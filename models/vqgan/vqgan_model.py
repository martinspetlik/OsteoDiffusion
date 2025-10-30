import torch
import pytorch_lightning as pl
import importlib

from models.vqgan.encoder_decoder import Encoder, Decoder
from models.vqgan.vector_quantizer import VectorQuantizer, EMAVectorQuantizer

from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler
from torch.amp import autocast


def get_obj_from_str(string, reload=False):
    """
    Dynamically import and return a class or function from a string path.
    :param string: Full module path, e.g. "torch.nn.Conv2d".
    :param reload: Whether to reload the module if already imported.
    :return: Imported Python object (class or function).
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate an object from a configuration dictionary.
    :param config: Dict with keys 'target' (full import path) and optional 'params'.
    :return: Instantiated Python object.
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    print(get_obj_from_str(config["target"]))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGAN(pl.LightningModule):
    """
    Vector-Quantized Generative Adversarial Network (VQGAN) implemented as a PyTorch LightningModule.
    Supports mixed precision, gradient accumulation, gradient clipping, EMA quantization, and
    optional generator freezing for staged training.
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
        Initialize the VQGAN model and its submodules.
        :param ed_config: Dictionary defining encoder/decoder architecture.
        :param loss_config: Configuration dict for the loss module.
        :param vq_config: Configuration dict for the vector quantizer.
        :param ckpt_path: Optional path to checkpoint for restoring model weights.
        :param ignore_keys: List of parameter prefixes to ignore during checkpoint load.
        :param monitor: Optional metric name for monitoring (e.g., val_loss).
        :param remap: Optional codebook remapping indices.
        :param sane_index_shape: Whether to enforce consistent code index shape.
        :param stage: Training stage indicator.
        :param accumulate_grad_batches: Number of steps to accumulate gradients.
        :param use_checkpoint: If True, use gradient checkpointing to save memory.
        :param quantizer_type: Either "EMA" or "Standard" quantizer type.
        :param generator_learning_rate: Learning rate for the generator.
        :param discriminator_learning_rate: Learning rate for the discriminator.
        :param gradient_clip: If True, clip gradients during training.
        :param freeze_generator: If True, freeze generator weights.
        :param disc_update_interval: Steps between discriminator updates.
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
        """
        Load model state from a checkpoint file.
        :param path: Path to checkpoint file (.ckpt or .pt).
        :param ignore_keys: List of parameter name prefixes to skip.
        :return: None
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
        Encode input tensor through encoder and quantizer.
        :param x: Input tensor of shape (B, C, D, H, W).
        :return: Tuple (quantized_tensor, embedding_loss, quantizer_info).
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
        Decode quantized representation to reconstruct the input image.
        :param quant: Quantized latent tensor.
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
        Decode a tensor of codebook indices into image space.
        :param code_b: Tensor of code indices.
        :return: Reconstructed image tensor.
        """
        quant_b = self.quantize.embed_code(code_b)
        return self.decode(quant_b)

    def forward(self, input, target=None):
        """
        Forward pass through encoder, quantizer, and decoder.
        :param input: Input image tensor.
        :param target: Optional target tensor (unused in this model).
        :return: Tuple (reconstruction, quantization_loss).
        """
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        """
        Extract input tensor from a batch dictionary.
        :param batch: Batch dictionary or tuple.
        :param k: Key to retrieve the tensor.
        :return: Float tensor.
        """
        return batch[k].float()

    @staticmethod
    def compute_grad_norm(model):
        """
        Compute L2 norm of all gradients in a model.
        :param model: PyTorch module with gradients.
        :return: Scalar gradient norm (float).
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def training_step(self, batch, batch_idx):
        """
        Perform one training iteration for generator and discriminator.
        Handles mixed precision, gradient accumulation, and logging.
        :param batch: Tuple (input_tensor, condition).
        :param batch_idx: Index of the current batch.
        :return: Dict with generator and discriminator losses.
        """
        with record_function("training_step"):
            x, cond = batch
            skip_pass = 1

            optimizer_g, optimizer_d = self.optimizers()

            # ======== Train generator (if not frozen) ========
            if not self.freeze_generator:
                self.toggle_optimizer(optimizer_g)
                with autocast('cuda'):
                    xrec, qloss = self(x)
                    aeloss, metrics_dict = self.loss(
                        qloss, x, xrec, optimizer_idx=0,
                        global_step=self.global_step,
                        last_layer=self.get_last_layer(),
                        skip_pass=skip_pass, split="train")

                # Log generator-related losses
                self.log("train/codebook_loss", metrics_dict["codebook_loss"], prog_bar=True)
                self.log("train/reconstruction_loss", metrics_dict["reconstruction_loss"], prog_bar=True)
                self.log("train/perceptual_loss", metrics_dict["perceptual_loss"], prog_bar=True)
                self.log("train/entropy_loss", metrics_dict["entropy_loss"], prog_bar=False)
                self.log("train/adversarial_generator_loss", metrics_dict["adversarial_generator_loss"], prog_bar=True)

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
                with torch.no_grad(), autocast('cuda'):
                    xrec, qloss = self(x)
                aeloss, metrics_dict = self.loss(
                    qloss, x, xrec, optimizer_idx=0,
                    global_step=self.global_step,
                    last_layer=self.get_last_layer(),
                    skip_pass=skip_pass, split="train", freeze_generator=True)

            self.log("train/generator_total_loss", metrics_dict["generator_total_loss"], prog_bar=True)

            # ======== Train discriminator ========
            discloss = torch.tensor(0.0, device=self.device)
            if (self.global_step % self.disc_update_interval) == 0:
                self.toggle_optimizer(optimizer_d)
                with autocast('cuda'):
                    discloss, metrics_dict = self.loss(
                        qloss, x, xrec, optimizer_idx=1,
                        global_step=self.global_step,
                        last_layer=self.get_last_layer(),
                        skip_pass=skip_pass, split="train")

                self.log("train/discriminator_total_loss", discloss, prog_bar=True)
                self.scaler.scale(discloss).backward()

                if self._grad_accum_step % self.accumulate_grad_batches == 0:
                    if self.gradient_clip:
                        if hasattr(self.loss, "discriminator"):
                            torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer_d)
                    self.scaler.update()
                    optimizer_d.zero_grad(set_to_none=True)
                self.untoggle_optimizer(optimizer_d)

            return {"gen_loss": aeloss, "d_loss": discloss}

    def compute_token_usage_entropy(self, usage_counts):
        """
        Compute codebook token usage entropy and normalized entropy.
        :param usage_counts: Tensor of shape [num_embeddings].
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
        Called at the beginning of each training epoch.
        Resets codebook usage statistics and stores generator weights if frozen.
        :return: None
        """
        self.quantize.reset_index_usage()
        if self.freeze_generator:
            self._initial_encoder_state = {name: param.clone().detach().cpu()
                                           for name, param in self.encoder.named_parameters()}

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Logs codebook utilization and entropy; checks generator weight integrity if frozen.
        :return: None
        """
        usage = self.quantize.index_usage_counts
        num_used = torch.count_nonzero(usage)
        usage_percent = 100.0 * num_used.item() / usage.numel()

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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Perform validation step (no gradient computation).
        :param batch: Tuple (input_tensor, condition).
        :param batch_idx: Batch index.
        :return: Dict with validation losses.
        """
        x, cond = batch
        skip_pass = 1

        with autocast('cuda'):
            xrec, qloss = self(x)
            aeloss, metrics_dict = self.loss(
                qloss, x, xrec, optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                skip_pass=skip_pass, split="val")

            discloss, disc_metrics = self.loss(
                qloss, x, xrec, optimizer_idx=1,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                skip_pass=skip_pass, split="val")

            self.log("val/generator_total_loss", metrics_dict["generator_total_loss"], prog_bar=True)
            self.log("val/discriminator_total_loss", discloss, prog_bar=False)

        return {"val_loss": aeloss, "val_disc_loss": discloss}

    def configure_optimizers(self):
        """
        Configure optimizers for generator and discriminator.
        :return: Tuple ([optimizer_g, optimizer_d], []).
        """
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.generator_learning_rate, betas=(0.5, 0.9)
        )

        if getattr(self.loss, "discriminator_2d", None) is not None and getattr(self.loss, "discriminator_3d", None) is not None:
            opt_disc = torch.optim.Adam(
                list(self.loss.discriminator_2d.parameters()) +
                list(self.loss.discriminator_3d.parameters()),
                lr=self.discriminator_learning_rate, betas=(0.5, 0.9))
        else:
            opt_disc = torch.optim.Adam(
                self.loss.discriminator.parameters(),
                lr=self.discriminator_learning_rate, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        Return the final convolution layer of the decoder (used for perceptual loss).
        :return: Weight tensor of the final conv layer.
        """
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        """
        Generate reconstructed images for visualization.
        :param batch: Input image batch tensor.
        :param kwargs: Additional logging arguments.
        :return: Dictionary with original and reconstructed images.
        """
        log = dict()
        x_src = batch[:1].float()
        xrec, _ = self(x_src)
        log["source"] = x_src
        log["recon"] = xrec
        return log

    def load_pretrained_generator_only(self, ckpt_path):
        """
        Load pretrained generator weights from checkpoint, skipping discriminator.
        :param ckpt_path: Path to checkpoint file.
        :return: None
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        generator_keys = [k for k in state_dict if k.startswith("encoder") or
                          k.startswith("decoder") or
                          k.startswith("quantize") or
                          k.startswith("quant_conv") or
                          k.startswith("post_quant_conv")]

        generator_state_dict = {k: state_dict[k] for k in generator_keys}
        missing, unexpected = self.load_state_dict(generator_state_dict, strict=False)

        print(f"Loaded generator weights from {ckpt_path}")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
