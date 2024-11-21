import torch
import torch.nn as nn
from models.components import SinusoidalPosEmb
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class UNet3DMedicalDiffusion(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        #rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        # def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
        #     dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        #
        # # realistically will not be able to generate that many frames of video... yet
        # self.time_rel_pos_bias = RelativePositionBias(
        #     heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        #assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size,
                                   init_kernel_size), padding=(0, init_padding, init_padding))

        # self.init_temporal_attn = Residual(
        #     PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        # self.has_cond = exists(cond_dim) or use_bert_text_cond
        # cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        #
        # self.null_cond_emb = nn.Parameter(
        #     torch.randn(1, cond_dim)) if self.has_cond else None
        #
        # cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        # block_klass = partial(ResnetBlock, groups=resnet_groups)
        # block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                # block_klass_cond(dim_in, dim_out),
                # block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        r = x.clone()

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'),
                               self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)




def default(val, default_val):
    return val if val is not None else default_val


class UNet3DWithTimestep(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32, time_emb_dim=64):
        super(UNet3DWithTimestep, self).__init__()

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_features)
        self.enc2 = self.conv_block(base_features, base_features * 2)
        self.enc3 = self.conv_block(base_features * 2, base_features * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_features * 4, base_features * 8)

        # Decoder
        self.dec3 = self.conv_block(base_features * 8, base_features * 4)
        self.dec2 = self.conv_block(base_features * 4, base_features * 2)
        self.dec1 = self.conv_block(base_features * 2, base_features)

        # Final Convolution
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Timestep fully connected layers to match encoder feature map sizes
        self.time_fc_enc1 = nn.Linear(time_emb_dim, base_features)
        self.time_fc_enc2 = nn.Linear(time_emb_dim, base_features * 2)
        self.time_fc_enc3 = nn.Linear(time_emb_dim, base_features * 4)
        self.time_fc_bottleneck = nn.Linear(time_emb_dim, base_features * 8)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # Get timestep embedding and pass through the embedding layer
        t_emb = self.time_embedding(t)

        # Apply timestep embeddings to each encoder block
        enc1 = self.enc1(x + self.time_fc_enc1(t_emb).view(-1, x.size(1), 1, 1, 1))
        enc2 = self.enc2(self.pool(enc1) + self.time_fc_enc2(t_emb).view(-1, enc1.size(1), 1, 1, 1))
        enc3 = self.enc3(self.pool(enc2) + self.time_fc_enc3(t_emb).view(-1, enc2.size(1), 1, 1, 1))

        # Bottleneck with timestep
        bottleneck = self.bottleneck(self.pool(enc3) + self.time_fc_bottleneck(t_emb).view(-1, enc3.size(1), 1, 1, 1))

        # Decoder with skip connections
        dec3 = self.upsample(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upsample(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upsample(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output
        return self.final_conv(dec1)



# Simple U-Net architecture for diffusion model
class SimpleUNet(nn.Module):
    def __init__(self, dim, in_channels=1,
                 out_channels=1,
                 init_conv_channels=8,
                 sinusoidal_pos_emb_theta=10000,
                 convnext_block_groups=8):
        super(SimpleUNet, self).__init__()

        self.dim = dim

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        time_dim = dim * 4

        #print("time dim ", time_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )


        self.init_conv = nn.Conv3d(in_channels, init_conv_channels, 3, padding=1)


        # self.encoder1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder1 = ConvNextBlock(
                            in_channels=init_conv_channels,
                            out_channels=32,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        )

        self.encoder2 = ConvNextBlock(
            in_channels=32,
            out_channels=64,
            time_embedding_dim=time_dim,
            group=convnext_block_groups,
        )

        self.mid_block1 = ConvNextBlock(64, 64, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(64, 64, time_embedding_dim=time_dim)

        self.decoder1 = ConvNextBlock(
            in_channels=128,
            out_channels=64,
            time_embedding_dim=time_dim,
            group=convnext_block_groups,
        )

        self.decoder2 = ConvNextBlock(
            in_channels=96,
            out_channels=init_conv_channels,
            time_embedding_dim=time_dim,
            group=convnext_block_groups,
        )

        # self.encoder2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        # self.decoder1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        # self.decoder2 = nn.Conv3d(64, out_channels, kernel_size=3, padding=1)

        # self.final_res_block = ConvNextBlock(in_channels, 4, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv3d(init_conv_channels, out_channels, 1)

    def forward(self, x, time):
        time_embedding = self.time_mlp(time)

        x = x.float()

        #print("x.shape ", x.shape)

        x = self.init_conv(x)
        r = x.clone()

        time_embedding = time_embedding.float()

        ex1 = self.encoder1(x, time_embedding)
        print("encoder1 x.shape ", ex1.shape)

        ex2 = self.encoder2(ex1, time_embedding)
        print("encoder2 ex2.shape ", ex2.shape)

        mx1 = self.mid_block1(ex2, time_embedding)
        mx2 = self.mid_block2(mx1, time_embedding)

        x = torch.cat((mx2, ex2), dim=1)
        print("cat mx2 ex2 shape ", x.shape)

        dx1 = self.decoder1(x, time_embedding)
        print("decoder1 x.shape ", dx1.shape)

        x = torch.cat((dx1, ex1), dim=1)

        dx2 = self.decoder2(x, time_embedding)
        #print("decoder2 x.shape ", dx2.shape)

        x = self.final_conv(dx2)
        exit()
        return x

        # x = torch.cat((x, r), dim=1)
        # x = self.final_res_block(x, time_embedding)
        #
        # return self.final_conv(x)

        # x1 = F.relu(self.encoder1(x))
        #
        # x2 = F.relu(self.encoder2(F.max_pool3d(x1, 2)))
        # x3 = F.relu(self.decoder1(F.interpolate(x2, scale_factor=2, mode='trilinear', align_corners=False)))
        # x4 = self.decoder2(x3)
        #return x


class UNet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            sinusoidal_pos_emb_theta=10000,
            convnext_block_groups=8,
            init_kernel_size=7
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        self.init_dim = default(init_dim, dim)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, self.init_dim, (1, init_kernel_size,
                                                        init_kernel_size), padding=(0, init_padding, init_padding))

        dims = [self.init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_in,
                            out_channels=dim_in,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        #print("self. downs ", self.downs)

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_embedding_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            in_channels=dim_out + dim_in,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        ConvNextBlock(
                            in_channels=dim_out,
                            out_channels=dim_out,
                            time_embedding_dim=time_dim,
                            group=convnext_block_groups,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ConvNextBlock(dim * 2, dim, time_embedding_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time):
        #print("==================== UNET ========================")
        #print("x.shape ", x.shape)

        x = x.float()
        # print("time.shape ", time.shape)
        #b, _, h, w = x.shape

        #batch, device = x.shape[0], x.device

        x = self.init_conv(x)
        #print("init conv x shape ", x.shape)
        r = x.clone()

        t = self.time_mlp(time)
        #print("time mlp shape ", t.shape)

        unet_stack = []
        for down1, down2, downsample in self.downs:
            x = down1(x, t)
            #print("down1 x shape ", x.shape)
            x = down2(x, t)
            #print("down2 x shape ", x.shape)
            #exit()
            unet_stack.append(x)
            #print("x.shape ", x.shape)
            x = downsample(x)
            #exit()

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)



        for up1, up2, upsample in self.ups:
            unet_stack_pop = unet_stack.pop()
            # print("x.shape ", x.shape)
            # print("unet_stack_pop.shape ", unet_stack_pop.shape)

            x = torch.cat((x, unet_stack_pop), dim=1)
            x = up1(x, t)
            x = up2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        return self.final_conv(x)


class DownSample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        #self.rearrange_layer = Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2)

        self.net = nn.Sequential(
            #Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2),
            nn.Conv3d(dim * 8, default(dim_out, dim), 1),
        )

    def forward(self, x):
        # print("x.shape ", x.shape)
        # y = self.rearrange_layer(x)
        #
        # print("y.shape ", y.shape)
        # exit()


        return self.net(x)

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(dim, dim_out or dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult=2,
        time_embedding_dim=None,
        norm=True,
        group=8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dim, in_channels))
            if time_embedding_dim
            else None
        )

        self.in_conv = nn.Conv3d(
            in_channels, in_channels, 7, padding=3, groups=in_channels
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
            nn.Conv3d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * mult),
            nn.Conv3d(out_channels * mult, out_channels, 3, padding=1),
        )

        self.residual_conv = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        h = self.in_conv(x)
        if self.mlp is not None and time_embedding is not None:
            assert self.mlp is not None, "MLP is None"
            h = h + rearrange(self.mlp(time_embedding), "b c -> b c 1 1 1")
        h = self.block(h)
        return h + self.residual_conv(x)


class ConvNextBlockSimplified(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult=2,
        time_embedding_dim=None,
        norm=True,
        group=8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dim, in_channels))
            if time_embedding_dim
            else None
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_conv = nn.Conv3d(
            in_channels, in_channels, 7, padding=3, groups=in_channels
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
            nn.Conv3d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * mult),
            nn.Conv3d(out_channels * mult, out_channels, 3, padding=1),
        )

        # self.block_groupnorm_1 = nn.GroupNorm(1, in_channels) if norm else nn.Identity()
        # self.block_conv_1 = nn.Conv3d(in_channels, out_channels * mult, 3, padding=1)
        # self.block_gelu = nn.GELU()
        # self.block_groupnorm_2 = nn.GroupNorm(1, out_channels * mult)
        # self.block_conv_2 = nn.Conv3d(out_channels * mult, out_channels, 3, padding=1)
        #
        # self.residual_conv = (
        #     nn.Conv3d(in_channels, out_channels, 1)
        #     if in_channels != out_channels
        #     else nn.Identity()
        # )

    def forward(self, x, time_embedding=None):
        verbose = False
        x = x.float()

        if verbose:
            print("before in conv x ", x)
            print("before in conv x shape ", x.shape)
            print("time embedding shape ", time_embedding.shape)
            print("in channels : {}, out channels: {}".format(self.in_channels, self.out_channels))
        h = self.in_conv(x)

        if verbose:
            print("in conv h shape ", h.shape)
            print("h.shape ", h.shape)

        if self.mlp is not None and time_embedding is not None:
            assert self.mlp is not None, "MLP is None"
            # print("self.mlp(time_embedding).shape ", self.mlp(time_embedding).shape)
            # print("rearrange(self.mlp(time_embedding).shape ", rearrange(self.mlp(time_embedding), "b c -> b c 1 1 1").shape)
            h = h + rearrange(self.mlp(time_embedding), "b c -> b c 1 1 1")

        if verbose:
            print("h shape before block ", h.shape)
            print("h ", h)



        # h = self.block_groupnorm_1(h)
        # print("block groupnorm 1 shape", h.shape)
        # h = self.block_conv_1(h)
        # print("block conv 1 shape ", h.shape)
        # h = self.block_groupnorm_2(h)
        # print("block groupnorm 2 shape ", h.shape)
        # h = self.block_conv_2(h)
        # print("block conv 2 shape ", h.shape)
        #
        # print("self.residual_conv(x) ", self.residual_conv(x).shape)
        #
        #
        h = self.block(h)



        return h
        # print("after block shape ", h.shape)
        # return h + self.residual_conv(x)


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
                    last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2),
                                          stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + res_channels, out_channels=in_channels // 2,
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=num_classes, kernel_size=(1, 1, 1))

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class UNet3DAmir(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3DAmir, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=out_channels,
                                      last_layer=True)

    def forward(self, input):
        # Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out