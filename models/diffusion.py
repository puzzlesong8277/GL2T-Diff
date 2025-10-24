import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import cv2
from einops.layers.torch import Rearrange

device = torch.device('cuda:0')

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class ZM_bn(nn.Module):
    def __init__(self):
        super(ZM_bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)

class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.
        
        Input: 
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        # 定义高斯模糊核的标准差 sigma，并计算每层的缩放因子 s_value
        sigma = 1.6
        s_value = 2 ** (1/3)
        # 生成不同尺度的高斯模糊核，并存储在 self.sigma_kernels 列表中
        self.sigma_kernels = [
            self.get_gaussian_kernel(2*i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float()#.to(device=device)

    def forward(self, x):
        G = x
        
        # Level 1
        L0 = Rearrange('b d h w -> b d (h w)')(G)
        L0_att= F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        L0_att = F.softmax(L0_att, dim=-1)
        
        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]
        
        for kernel in self.sigma_kernels:
            kernel=kernel.to (device=x.device)
            G=G.to (device=x.device)
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels).to(device = x.device)
            pyramid.append(G)
        
        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = Rearrange('b d h w -> b d (h w)')(L)
            L_att= F.softmax(L, dim=2) @ L.transpose(1, 2) 
            attention_maps.append(L_att)

        return sum(attention_maps)
# class EfficientFrequencyAttention(nn.Module):
#     """
#     args:
#         in_channels:    (int) : Embedding Dimension.
#         key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
#         value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2). 
#         pyramid_levels  (int) : Number of pyramid levels.
#     input:
#         x : [B, D, H, W]
#     output:
#         Efficient Attention : [B, D, H, W]
    
#     """
    
#     def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3):
#         super().__init__()
#         self.in_channels = in_channels
#         self.key_channels = key_channels
#         self.value_channels = value_channels

#         self.keys = nn.Conv2d(in_channels, key_channels, 1) 
#         self.queries = nn.Conv2d(in_channels, key_channels, 1)
#         self.values = nn.Conv2d(in_channels, value_channels, 1)
#         self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
#         # Build a laplacian pyramid
#         self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels) 
        
#         self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)
                
        
#     def forward(self, x):
#         n, _, h, w = x.size()
        
#         # Efficient Attention
#         keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
#         queries = F.softmax(self.queries(x).reshape(n, self.key_channels, h * w), dim=1)
#         values = self.values(x).reshape((n, self.value_channels, h * w))          
#         context = keys @ values.transpose(1, 2) # dk*dv            
#         attended_value = (context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w) # n*dv
#         eff_attention  = self.reprojection(attended_value)

#         # Freqency Attention
#         freq_context = self.freq_attention(x)
#         freq_attention =  (freq_context.transpose(1, 2) @ queries).reshape(n, self.value_channels , h, w) 
        
#         # Attention Aggregation: Efficient Frequency Attention (EF-Att) Block
#         attention = torch.cat([eff_attention[:, :, None, ...], freq_attention[:, :, None, ...]], dim=2)
#         attention = self.conv_dw(attention)[:, :, 0, ...] 

#         return attention
class FFTInteraction_N(nn.Module):
    def __init__(self, in_nc,out_nc):
        super(FFTInteraction_N,self).__init__()
        self.post = nn.Conv2d(in_nc,out_nc,1,1,0)
        self.mid = nn.Conv2d(in_nc,in_nc,3,1,1,groups=in_nc)
        self.conv1 = nn.Conv2d(in_nc,out_nc,3,1,1,groups=in_nc)


    def forward(self,x_enc,x_dec):
        x_enc = torch.fft.rfft2(x_enc, norm='backward')
        x_dec = torch.fft.rfft2(x_dec, norm='backward')
        x_freq_amp = torch.abs(x_enc)
        x_freq_pha = torch.angle(x_dec)
        x_freq_pha = self.mid(x_freq_pha)
        real = x_freq_amp * torch.cos(x_freq_pha)
        real = self.post(real)                              
        imag = x_freq_amp * torch.sin(x_freq_pha)
        imag = self.conv1(imag)
        x_recom = torch.complex(real, imag)
        x_recom = torch.fft.irfft2(x_recom)

        return x_recom
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Sequential(torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0),torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,groups=in_channels))
        self.k = nn.Sequential(torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0),torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,groups=in_channels))
        self.v = nn.Sequential(torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0),torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,groups=in_channels))
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=3) 
        self.fusion=FFTInteraction_N(in_nc=in_channels,out_nc=in_channels)
        self.bn = ZM_bn()
        # self.fusion=nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels,kernel_size=1,padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        queries=q
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        # print(h_.shape,'h')
        # Freqency Attention
        freq_context = self.freq_attention(x)#(4,512,512)   x.shape(4,512,16,16)

        freq_attention =  (freq_context.transpose(1, 2) @ queries).reshape(b, c , h, w) 
        # print(freq_attention.shape,'freq')
        fusion_feature=self.fusion(h_ ,freq_attention)
        return (x+fusion_feature)+self.bn(x+fusion_feature)
        # ssreturn F.gelu(self.fusion(torch.cat([x,freq_attention],dim=1)))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,#3
                                       self.ch,#128
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
