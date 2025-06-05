import random
import torch
import pytorch_lightning as pl
import importlib

from models.vqgan.adopted_codes.VQGAN_components import Encoder, Decoder
from models.cnn_diffusion.aldm.quantize import VectorQuantizer2 as VectorQuantizer
# from models.cnn_diffusion.aldm.quantize import GumbelQuantize
# from models.cnn_diffusion.aldm.quantize import EMAVectorQuantizer

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

class ALDMVQGAN(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 modalities=[],
                 #image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 stage=1,
                 ):
        super().__init__()
        self._name = "ALDMVQGAN"
        #self.image_key = image_key
        self.automatic_optimization = False
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.modalities = modalities
        self.stage = stage

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

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
        x = batch
        B, C, T, H, W = x.shape
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
        x = batch
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/recon_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
        source = random.choice(self.modalities)
        target = random.choice(self.modalities)
        x_src = self.get_input(batch, source).to(self.device)
        x_tar = self.get_input(batch, target).to(self.device)
        xrec, _ = self(x_src)
        if x_src.shape[1] > 3:
            x_src = self.to_rgb(x_src)
            xrec = self.to_rgb(xrec)
        log["source"] = x_src
        log["target"] = x_tar
        log["recon"] = xrec
        return log

    # def to_rgb(self, x):
    #     assert self.image_key == "segmentation"
    #     if not hasattr(self, "colorize"):
    #         self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1))
    #     x = F.conv2d(x, weight=self.colorize)
    #     x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    #     return x
