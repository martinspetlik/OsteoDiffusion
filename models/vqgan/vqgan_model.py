import torch
import pytorch_lightning as pl
import importlib

from models.vqgan.encoder_decoder import Encoder, Decoder
from models.vqgan.vector_quantizer import VectorQuantizer


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
                 #image_key="image",
                 #colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 stage=1,
                 ):
        super().__init__()
        self._name = "VQGAN"
        #self.image_key = image_key
        self.automatic_optimization = False

        self.encoder = Encoder(**ed_config)
        self.decoder = Decoder(**ed_config)
        self.quantize = VectorQuantizer(**vq_config,
                                        remap=remap, sane_index_shape=sane_index_shape)

        self.loss = instantiate_from_config(loss_config)

        self.quant_conv = torch.nn.Conv3d(ed_config["z_channels"], vq_config["dim_embed"], 1)
        self.post_quant_conv = torch.nn.Conv3d(vq_config["dim_embed"], ed_config["z_channels"], 1)
        self.stage = stage

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

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
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        return self.decode(quant_b)

    def forward(self, input, target=None):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        return batch[k].float()

    def training_step(self, batch, batch_idx):
        x, cond = batch
        skip_pass = 1

        xrec, qloss = self(x)

        optimizer_g, optimizer_d = self.optimizers()

        # Train generator
        self.toggle_optimizer(optimizer_g)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx=0,
                                        global_step=self.global_step,
                                        last_layer=self.get_last_layer(),
                                        skip_pass=skip_pass, split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/recon_loss", log_dict_ae["train/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae)
        self.manual_backward(aeloss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx=1,
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(),
                                            skip_pass=skip_pass, split="train")

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc)
        self.manual_backward(discloss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        return {"gen_loss": aeloss, "d_loss": discloss}

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/recon_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        for p in self.encoder.parameters(): p.requires_grad = True
        for p in self.decoder.parameters(): p.requires_grad = True
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        #source = random.choice(self.modalities)
        #target = random.choice(self.modalities)
        x_src = batch[:1].float().to(self.device)
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

