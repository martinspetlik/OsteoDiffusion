import torch
import pytorch_lightning as pl
import importlib

from models.vqgan.encoder_decoder import Encoder, Decoder
from models.vqgan.vector_quantizer import VectorQuantizer, EMAVectorQuantizer

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.checkpoint import checkpoint

from torch.cuda.amp import GradScaler
from torch.amp import autocast


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    print(get_obj_from_str(config["target"]))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGAN(pl.LightningModule):
    def __init__(self,
                 ed_config,
                 loss_config,
                 vq_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 stage=1,
                 accumulate_grad_batches=4,
                 use_checkpoint=False,
                 quantizer_type="EMA",
                 generator_learning_rate=1e-5,
                 discriminator_learning_rate=2e-4,
                 gradient_clip=False,
                 freeze_generator=False):
        super().__init__()
        self._name = "VQGAN"
        #self.image_key = image_key
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.use_checkpoint = use_checkpoint
        self._grad_accum_step = 0
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.gradient_clip = gradient_clip
        self.freeze_generator = freeze_generator

        self.encoder = Encoder(**ed_config)
        self.decoder = Decoder(**ed_config)

        if quantizer_type == "EMA":
            self.quantize = EMAVectorQuantizer(**vq_config,
                                            remap=remap, sane_index_shape=sane_index_shape)

        else:
            self.quantize = VectorQuantizer(**vq_config,
                                            remap=remap, sane_index_shape=sane_index_shape)

        print("quantizer ", self.quantize)

        self.loss = instantiate_from_config(loss_config)

        self.quant_conv = torch.nn.Conv3d(ed_config["z_channels"], vq_config["dim_embed"], 1)
        self.post_quant_conv = torch.nn.Conv3d(vq_config["dim_embed"], ed_config["z_channels"], 1)
        self.stage = stage

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

        self.scaler = GradScaler()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        print("use checkpoint ", self.use_checkpoint)
        if self.use_checkpoint and self.training:
            def run_encoder(x_in):
                h = self.encoder(x_in)
                h = self.quant_conv(h)
                return h
            h = checkpoint(run_encoder, x)
        else:
            h = self.encoder(x)
            h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        if self.use_checkpoint and self.training:
            dec = checkpoint(self.decoder, quant)
        else:
            dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        return self.decode(quant_b)

    def forward(self, input, target=None):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        return batch[k].float()

    @staticmethod
    def compute_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def training_step(self, batch, batch_idx):
        with record_function("training_step"):
            x, cond = batch
            skip_pass = 1

            optimizer_g, optimizer_d = self.optimizers()

            # ======== Train generator (only if not frozen) ========
            if not self.freeze_generator:
                self.toggle_optimizer(optimizer_g)
                with autocast('cuda'):
                    xrec, qloss = self(x)
                    aeloss, metrics_dict = self.loss(
                        qloss, x, xrec, optimizer_idx=0,
                        global_step=self.global_step,
                        last_layer=self.get_last_layer(),
                        skip_pass=skip_pass, split="train")

                self.log("train/codebook_loss", metrics_dict["codebook_loss"], prog_bar=True, logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/reconstruction_loss", metrics_dict["reconstruction_loss"], prog_bar=True,
                         logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/perceptual_loss", metrics_dict["perceptual_loss"], prog_bar=True, logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/entropy_loss", metrics_dict["entropy_loss"], prog_bar=False, logger=True,
                         on_step=True,
                         on_epoch=True)
                self.log("train/adversarial_generator_loss", metrics_dict["adversarial_generator_loss"],
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/g_2d_loss", metrics_dict["g_2d_loss"],
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/g_3d_loss", metrics_dict["g_3d_loss"],
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/disc_factor", metrics_dict["disc_factor"],
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.scaler.scale(aeloss).backward()
                self._grad_accum_step += 1

                if self._grad_accum_step % self.accumulate_grad_batches == 0:
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
                    self.scaler.step(optimizer_g)
                    self.scaler.update()
                    optimizer_g.zero_grad(set_to_none=True)
                self.untoggle_optimizer(optimizer_g)

            else:
                # Just run generator forward pass without gradients (so xrec is still available)
                with torch.no_grad(), autocast('cuda'):
                    xrec, qloss = self(x)
                aeloss, metrics_dict = self.loss(
                    qloss, x, xrec, optimizer_idx=0,
                    global_step=self.global_step,
                    last_layer=self.get_last_layer(),
                    skip_pass=skip_pass, split="train", freeze_generator=True)

            self.log("train/generator_total_loss", metrics_dict["generator_total_loss"],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # ======== Train discriminator (inside self.loss) ========
            self.toggle_optimizer(optimizer_d)
            with autocast('cuda'):
                print("training disc autocast ", torch.is_autocast_enabled())
                discloss, metrics_dict = self.loss(qloss, x, xrec, optimizer_idx=1,
                                                    global_step=self.global_step,
                                                    last_layer=self.get_last_layer(),
                                                    skip_pass=skip_pass, split="train")
            self.log("train/discriminator_total_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_2d_real", metrics_dict["logits_2d_real"], prog_bar=True, logger=True, on_step=True,on_epoch=True)
            self.log("train/logits_2d_fake",  metrics_dict["logits_2d_fake"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_3d_real", metrics_dict["logits_3d_real"], prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
            self.log("train/logits_3d_fake", metrics_dict["logits_2d_fake"], prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)

            self.scaler.scale(discloss).backward()

            # Log discriminator gradient norm
            if self.loss.discriminator_2d is not None and self.loss.discriminator_3d is not None:
                grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator_2d)
                self.log("train/discriminator_grad_norm_2d", grad_norm, on_step=True, on_epoch=False, prog_bar=True,
                         logger=True)

                grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator_3d)
                self.log("train/discriminator_grad_norm_3d", grad_norm, on_step=True, on_epoch=False, prog_bar=True,
                         logger=True)
            else:
                grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator)
                self.log("train/discriminator_grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=True,
                         logger=True)

            if self._grad_accum_step % self.accumulate_grad_batches == 0:
                # Clip gradients for the discriminator inside self.loss
                if self.gradient_clip:
                    if hasattr(self.loss, "discriminator"):
                        torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), max_norm=1.0)
                    else:
                        print("No discriminator found in self.loss. Skipping gradient clipping.")
                self.scaler.step(optimizer_d)
                self.scaler.update()
                optimizer_d.zero_grad(set_to_none=True)
            self.untoggle_optimizer(optimizer_d)

            return {"gen_loss": aeloss, "d_loss": discloss}

    def compute_token_usage_entropy(self, usage_counts):
        """
        usage_counts: torch.Tensor of shape [num_embeddings], dtype=int
        Returns: (entropy, normalized_entropy)
        """
        counts = usage_counts.float()
        probs = counts / counts.sum()
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log2(probs[mask]))

        max_entropy = torch.log2(torch.tensor(len(usage_counts), dtype=probs.dtype))
        normalized_entropy = entropy / max_entropy

        return entropy.item(), normalized_entropy.item()

    def on_train_epoch_start(self):
        self.quantize.reset_index_usage()

        if self.freeze_generator:
            self._initial_encoder_state = {name: param.clone().detach().cpu()
                                           for name, param in self.encoder.named_parameters()}

    def on_train_epoch_end(self):
        usage = self.quantize.index_usage_counts  # Tensor shape: [num_embeddings]
        num_used = torch.count_nonzero(usage)
        usage_percent = 100.0 * num_used.item() / usage.numel()
        # self.log("train/used_codebook_percent", usage_percent)
        # self.log("train/used_codebook_count", num_used)

        # Compute entropy
        entropy, norm_entropy = self.compute_token_usage_entropy(usage)

        self.log("train/used_codebook_percent", usage_percent)
        self.log("train/used_codebook_count", num_used)
        self.log("train/codebook_entropy", entropy)
        self.log("train/codebook_entropy_norm", norm_entropy)

        if self.freeze_generator:
            for name, param in self.encoder.named_parameters():
                initial = self._initial_encoder_state[name]
                current = param.detach().cpu()
                if not torch.allclose(initial, current, atol=1e-6):
                    print(f"[WARNING] Generator parameter '{name}' has changed!")
                else:
                    print(f"[OK] Generator parameter '{name}' is unchanged.")

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        with autocast('cuda'):
            print("validation autocast ", torch.is_autocast_enabled())
            xrec, qloss = self(x)
            aeloss, metrics_dict = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

            # Rename log keys here to match the internal log dict keys
            self.log("val/generator_total_loss", metrics_dict["generator_total_loss"], prog_bar=True, logger=False,
                     on_step=True, on_epoch=True)
            # Log individual loss components from the dict with consistent naming
            self.log("val/codebook_loss", metrics_dict["codebook_loss"], prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log("val/reconstruction_loss", metrics_dict["reconstruction_loss"], prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log("val/perceptual_loss", metrics_dict["perceptual_loss"], prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log("val/entropy_loss", metrics_dict["entropy_loss"], prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
            self.log("val/adversarial_generator_loss", metrics_dict["adversarial_generator_loss"],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)


            discloss, metrics_dict = self.loss(qloss, x, xrec, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            self.log("val/discriminator_total_loss", discloss, prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
            self.log("val/logits_real", metrics_dict["logits_real"], prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)
            self.log("val/logits_fake", metrics_dict["logits_fake"], prog_bar=True, logger=True, on_step=True,
                     on_epoch=True)

        #rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/recon_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)

        return metrics_dict["generator_total_loss"]

    def configure_optimizers(self):
        for p in self.encoder.parameters(): p.requires_grad = True
        for p in self.decoder.parameters(): p.requires_grad = True
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.generator_learning_rate, betas=(0.5, 0.9)
        )

        if self.loss.discriminator_2d is not None and self.loss.discriminator_3d is not None:
            opt_disc = torch.optim.Adam(list(self.loss.discriminator_2d.parameters()) + list(self.loss.discriminator_3d.parameters()), lr=self.discriminator_learning_rate,
                                        betas=(0.5, 0.9))
        else:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=self.discriminator_learning_rate, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        #source = random.choice(self.modalities)
        #target = random.choice(self.modalities)
        x_src = batch[:1].float()#.to(self.device)
        #x_tar = batch[:1].float().to(self.device)
        print("x_src.shape ", x_src.shape)
        xrec, _ = self(x_src)

        #print("x_tar.shape ", x_tar.shape)
        # if x_src.shape[1] > 3:
        #     x_src = self.to_rgb(x_src)
        #     xrec = self.to_rgb(xrec)
        log["source"] = x_src
        #log["target"] = x_tar
        log["recon"] = xrec
        return log

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     if batch_idx >= self.max_batches:
    #         return
    #
    #     x, _ = batch  # assuming (x, y)
    #     x = x.detach().cpu().numpy()
    #     y_hat = outputs.detach().cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs['pred'].detach().cpu().numpy()
    #
    #     np.save(os.path.join(self.save_dir, f"epoch{trainer.current_epoch}_batch{batch_idx}_input.npy"), x)
    #     np.save(os.path.join(self.save_dir, f"epoch{trainer.current_epoch}_batch{batch_idx}_output.npy"), y_hat)

