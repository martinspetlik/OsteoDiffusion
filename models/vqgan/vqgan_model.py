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
    """
    Dynamically import an object (class or function) from a string path.

    :param string: Fully qualified class or function path, e.g. 'torch.nn.Conv3d'
    :param reload: If True, force reload of the module.
    :return: Imported Python object (class/function).
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate a Python object from a configuration dictionary.

    :param config: Dictionary with at least a 'target' key specifying the import path.
    :return: Instantiated object.
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    print(get_obj_from_str(config["target"]))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGAN(pl.LightningModule):
    """
    Vector-Quantized Generative Adversarial Network (VQGAN) implemented as a LightningModule.
    Combines an encoder-decoder autoencoder, vector quantizer, and adversarial loss.
    """

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
                 freeze_generator=False,
                 disc_update_interval=1):
        """
        Initialize the VQGAN architecture and training parameters.

        :param ed_config: Encoder/decoder architecture configuration dictionary.
        :param loss_config: Loss function configuration dictionary.
        :param vq_config: Vector quantizer configuration dictionary.
        :param ckpt_path: Optional checkpoint path for model initialization.
        :param ignore_keys: Keys to ignore when loading state dict from checkpoint.
        :param monitor: Metric name for monitoring during training.
        :param remap: Optional embedding remap for quantizer.
        :param sane_index_shape: Whether to reshape index tensors consistently.
        :param stage: Training stage (for multi-stage training setups).
        :param accumulate_grad_batches: Number of batches for gradient accumulation.
        :param use_checkpoint: Whether to use gradient checkpointing for memory savings.
        :param quantizer_type: Type of vector quantizer ("EMA" or standard).
        :param generator_learning_rate: Learning rate for the generator optimizer.
        :param discriminator_learning_rate: Learning rate for the discriminator optimizer.
        :param gradient_clip: Whether to apply gradient clipping.
        :param freeze_generator: If True, freeze generator weights during training.
        :param disc_update_interval: Number of generator steps per discriminator update.
        """
        super().__init__()
        self._name = "VQGAN"
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.use_checkpoint = use_checkpoint
        self._grad_accum_step = 0
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.gradient_clip = gradient_clip
        self.freeze_generator = freeze_generator
        self.disc_update_interval = disc_update_interval

        # Initialize encoder and decoder networks
        self.encoder = Encoder(**ed_config)
        self.decoder = Decoder(**ed_config)

        # Select vector quantizer type
        if quantizer_type == "EMA":
            self.quantize = EMAVectorQuantizer(**vq_config,
                                               remap=remap, sane_index_shape=sane_index_shape)
        else:
            self.quantize = VectorQuantizer(**vq_config,
                                            remap=remap, sane_index_shape=sane_index_shape)

        print("quantizer ", self.quantize)

        # Loss function instantiation
        self.loss = instantiate_from_config(loss_config)

        # 1x1 convolutions for latent space transformation
        self.quant_conv = torch.nn.Conv3d(ed_config["z_channels"], vq_config["dim_embed"], 1)
        self.post_quant_conv = torch.nn.Conv3d(vq_config["dim_embed"], ed_config["z_channels"], 1)
        self.stage = stage

        # Load pretrained checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor

        # Automatic mixed precision scaler
        self.scaler = GradScaler()

    def init_from_ckpt(self, path, ignore_keys=list()):
        """
        Load model weights from a checkpoint file.

        :param path: Path to checkpoint file.
        :param ignore_keys: List of keys to skip when loading weights.
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        Encode input tensor into quantized latent representation.

        :param x: Input tensor (B, C, D, H, W).
        :return: Tuple (quantized output, embedding loss, quantizer info).
        """
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
        """
        Decode quantized latent tensor back into image space.

        :param quant: Quantized tensor from the encoder.
        :return: Reconstructed image tensor.
        """
        quant = self.post_quant_conv(quant)
        if self.use_checkpoint and self.training:
            dec = checkpoint(self.decoder, quant)
        else:
            dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """
        Decode integer code indices to image.

        :param code_b: Tensor of codebook indices.
        :return: Reconstructed image tensor.
        """
        quant_b = self.quantize.embed_code(code_b)
        return self.decode(quant_b)

    def forward(self, input, target=None):
        """
        Forward pass through the VQGAN generator.

        :param input: Input image tensor.
        :param target: Optional target (unused in forward).
        :return: Tuple (reconstructed image, quantization loss).
        """
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        """
        Extract input tensor from batch dictionary.

        :param batch: Dictionary or tuple containing data tensors.
        :param k: Key for the tensor to retrieve.
        :return: Float tensor.
        """
        return batch[k].float()

    @staticmethod
    def compute_grad_norm(model):
        """
        Compute total L2 norm of gradients for a model.

        :param model: PyTorch model.
        :return: Scalar representing gradient norm.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def training_step(self, batch, batch_idx):
        """
        Perform one training step, updating generator and discriminator.

        Includes mixed precision, gradient accumulation, clipping, and optional freezing.

        :param batch: Tuple (images, conditioning info).
        :param batch_idx: Index of the current batch.
        :return: Dict containing generator and discriminator losses.
        """
        with record_function("training_step"):
            x, cond = batch
            skip_pass = 1
            optimizer_g, optimizer_d = self.optimizers()

            # === Generator training ===
            if not self.freeze_generator:
                self.toggle_optimizer(optimizer_g)
                with autocast('cuda'):
                    xrec, qloss = self(x)
                    aeloss, metrics_dict = self.loss(
                        qloss, x, xrec, optimizer_idx=0,
                        global_step=self.global_step,
                        last_layer=self.get_last_layer(),
                        skip_pass=skip_pass, split="train")

                # Log generator metrics
                self.log("train/codebook_loss", metrics_dict["codebook_loss"], prog_bar=True, logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/reconstruction_loss", metrics_dict["reconstruction_loss"], prog_bar=True,
                         logger=True, on_step=True, on_epoch=True)
                self.log("train/perceptual_loss", metrics_dict["perceptual_loss"], prog_bar=True,
                         logger=True, on_step=True, on_epoch=True)
                self.log("train/entropy_loss", metrics_dict["entropy_loss"], prog_bar=False,
                         logger=True, on_step=True, on_epoch=True)
                self.log("train/adversarial_generator_loss", metrics_dict["adversarial_generator_loss"],
                         prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/g_2d_loss", metrics_dict["g_2d_loss"], prog_bar=True, logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/g_3d_loss", metrics_dict["g_3d_loss"], prog_bar=True, logger=True,
                         on_step=True, on_epoch=True)
                self.log("train/disc_factor", metrics_dict["disc_factor"], prog_bar=True,
                         logger=True, on_step=True, on_epoch=True)

                # Backpropagation with gradient accumulation
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
                # Skip generator training but still produce reconstruction
                with torch.no_grad(), autocast('cuda'):
                    xrec, qloss = self(x)
                aeloss, metrics_dict = self.loss(
                    qloss, x, xrec, optimizer_idx=0,
                    global_step=self.global_step,
                    last_layer=self.get_last_layer(),
                    skip_pass=skip_pass, split="train", freeze_generator=True)

            self.log("train/generator_total_loss", metrics_dict["generator_total_loss"],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)

            # === Discriminator training ===
            discloss = torch.tensor(0.0, device=self.device)
            if (self.global_step % self.disc_update_interval) == 0:
                self.toggle_optimizer(optimizer_d)
                with autocast('cuda'):
                    discloss, metrics_dict = self.loss(
                        qloss, x, xrec, optimizer_idx=1,
                        global_step=self.global_step,
                        last_layer=self.get_last_layer(),
                        skip_pass=skip_pass, split="train")

                # Log discriminator metrics
                self.log("train/discriminator_total_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/logits_2d_real", metrics_dict["logits_2d_real"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/logits_2d_fake", metrics_dict["logits_2d_fake"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/logits_3d_real", metrics_dict["logits_3d_real"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log("train/logits_3d_fake", metrics_dict["logits_2d_fake"], prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.scaler.scale(discloss).backward()

                # Log gradient norms for discriminator(s)
                if self.loss.discriminator_2d is not None and self.loss.discriminator_3d is not None:
                    grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator_2d)
                    self.log("train/discriminator_grad_norm_2d", grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                    grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator_3d)
                    self.log("train/discriminator_grad_norm_3d", grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                else:
                    grad_norm = VQGAN.compute_grad_norm(self.loss.discriminator)
                    self.log("train/discriminator_grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)

                if self._grad_accum_step % self.accumulate_grad_batches == 0:
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
        Compute entropy and normalized entropy for codebook token usage.

        :param usage_counts: Tensor of shape [num_embeddings], integer usage counts per code.
        :return: Tuple (entropy, normalized_entropy).
        """
        counts = usage_counts.float()
        probs = counts / counts.sum()
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log2(probs[mask]))

        max_entropy = torch.log2(torch.tensor(len(usage_counts), dtype=probs.dtype))
        normalized_entropy = entropy / max_entropy

        return entropy.item(), normalized_entropy.item()

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.

        Resets quantizer code usage statistics and, if the generator is frozen,
        stores a snapshot of the encoder weights for later consistency verification.
        """
        self.quantize.reset_index_usage()

        if self.freeze_generator:
            self._initial_encoder_state = {
                name: param.clone().detach().cpu()
                for name, param in self.encoder.named_parameters()
            }

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.

        Logs codebook utilization statistics (usage %, entropy).
        If the generator is frozen, verifies encoder parameters remain unchanged.
        """
        usage = self.quantize.index_usage_counts  # Tensor of shape [num_embeddings]
        num_used = torch.count_nonzero(usage)
        usage_percent = 100.0 * num_used.item() / usage.numel()

        # Compute entropy and normalized entropy
        entropy, norm_entropy = self.compute_token_usage_entropy(usage)

        # Log metrics
        self.log("train/used_codebook_percent", usage_percent)
        self.log("train/used_codebook_count", num_used)
        self.log("train/codebook_entropy", entropy)
        self.log("train/codebook_entropy_norm", norm_entropy)

        # Verify frozen generator weights have not changed
        if self.freeze_generator:
            for name, param in self.encoder.named_parameters():
                initial = self._initial_encoder_state[name]
                current = param.detach().cpu()
                if not torch.allclose(initial, current, atol=1e-6):
                    print(f"[WARNING] Generator parameter '{name}' has changed!")
                else:
                    print(f"[OK] Generator parameter '{name}' is unchanged.")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step without gradient computation.

        Evaluates generator and discriminator losses and logs all metrics.

        :param batch: Tuple (image_tensor, conditioning_info)
        :param batch_idx: Index of batch in validation set
        :return: Dict containing validation losses for generator and discriminator.
        """
        x, cond = batch
        skip_pass = 1

        with autocast('cuda'):
            # === Generator evaluation ===
            xrec, qloss = self(x)
            aeloss, metrics_dict = self.loss(
                qloss, x, xrec, optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                skip_pass=skip_pass, split="val"
            )
            print("Validation step is running!")  # Debug info

            # Log generator metrics
            self.log("val/generator_total_loss", metrics_dict["generator_total_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/reconstruction_loss", metrics_dict["reconstruction_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/perceptual_loss", metrics_dict["perceptual_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/codebook_loss", metrics_dict["codebook_loss"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/entropy_loss", metrics_dict["entropy_loss"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/g_2d_loss", metrics_dict["g_2d_loss"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/g_3d_loss", metrics_dict["g_3d_loss"], prog_bar=False, on_step=False, on_epoch=True)

            # === Discriminator evaluation ===
            discloss, disc_metrics = self.loss(
                qloss, x, xrec, optimizer_idx=1,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                skip_pass=skip_pass,
                split="val"
            )

            # Log discriminator metrics
            self.log("val/discriminator_total_loss", discloss, prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/logits_2d_real", disc_metrics["logits_2d_real"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/logits_2d_fake", disc_metrics["logits_2d_fake"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/logits_3d_real", disc_metrics["logits_3d_real"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("val/logits_3d_fake", disc_metrics["logits_3d_fake"], prog_bar=False, on_step=False, on_epoch=True)

        return {"val_loss": aeloss, "val_disc_loss": discloss}

    def configure_optimizers(self):
        """
        Configure separate Adam optimizers for generator and discriminator.

        :return: Tuple ([generator_optimizer, discriminator_optimizer], [])
        """
        # Enable gradient computation
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

        # === Generator optimizer ===
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.generator_learning_rate, betas=(0.5, 0.9)
        )

        # === Discriminator optimizer ===
        if self.loss.discriminator_2d is not None and self.loss.discriminator_3d is not None:
            opt_disc = torch.optim.Adam(
                list(self.loss.discriminator_2d.parameters()) +
                list(self.loss.discriminator_3d.parameters()),
                lr=self.discriminator_learning_rate, betas=(0.5, 0.9)
            )
        else:
            opt_disc = torch.optim.Adam(
                self.loss.discriminator.parameters(),
                lr=self.discriminator_learning_rate, betas=(0.5, 0.9)
            )

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        Get the final convolution layer of the decoder.

        Used for perceptual loss weighting in adversarial training.

        :return: Decoder output convolution weight tensor.
        """
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        """
        Generate reconstructions for visualization and logging.

        :param batch: Input batch tensor.
        :param kwargs: Additional arguments (unused).
        :return: Dict with 'source' and 'recon' tensors for visualization.
        """
        log = dict()
        x_src = batch[:1].float()
        print("x_src.shape ", x_src.shape)
        xrec, _ = self(x_src)
        log["source"] = x_src
        log["recon"] = xrec
        return log

    def load_pretrained_generator_only(self, ckpt_path):
        """
        Load only generator (encoder, decoder, quantizer) weights from checkpoint.

        :param ckpt_path: Path to pretrained model checkpoint.
        :return: None
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Filter generator-related keys only
        generator_keys = [
            k for k in state_dict if k.startswith("encoder") or
            k.startswith("decoder") or
            k.startswith("quantize") or
            k.startswith("quant_conv") or
            k.startswith("post_quant_conv")
        ]

        generator_state_dict = {k: state_dict[k] for k in generator_keys}
        missing, unexpected = self.load_state_dict(generator_state_dict, strict=False)

        print(f"Loaded generator weights from {ckpt_path}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

