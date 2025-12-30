import sys
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import repeat
from functools import partial
import torch.utils.checkpoint as checkpoint

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    raise ImportError("请安装 einops: pip install einops")

def to_2tuple(x):
    if isinstance(x, numbers.Number):
        return (x, x)
    return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        if x.dim() == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): return x

class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1, groups=hidden_features), 
            nn.GELU())
        self.hidden_features = hidden_features
    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.before_add = emptyModule()
        self.after_add = emptyModule()
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.before_add(x)
        x = x + self.dwconv(x, x_size)
        x = self.after_add(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class PSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.permuted_window_size = (window_size[0] // 2, window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='xy'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)
        aligned_relative_position_index = aligned_relative_position_index.reshape(
            self.permuted_window_size[0], self.permuted_window_size[1], 1, 1, self.permuted_window_size[0] * self.permuted_window_size[1]
        ).repeat(1, 1, 2, 2, 1).permute(0, 2, 1, 3, 4).reshape(
            4 * self.permuted_window_size[0] * self.permuted_window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1]
        )
        self.register_buffer('aligned_relative_position_index', aligned_relative_position_index)
        
        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        kv = self.kv(x).reshape(b_, self.permuted_window_size[0], 2, self.permuted_window_size[1], 2, 2, c // 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, n // 4, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n // 4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n // 4)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def flops(self, n):
        flops = 0
        flops += n * self.dim * 1.5 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n/4
        flops += self.num_heads * n * n/4 * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops

class PSA_Block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.permuted_window_size = window_size // 2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)

        self.attn = PSA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)
        self.after_norm1 = emptyModule()
        self.after_attention = emptyModule()
        self.residual_after_attention = emptyModule()
        self.after_norm2 = emptyModule()
        self.after_mlp = emptyModule()
        self.residual_after_mlp = emptyModule()

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        permuted_window_mask = torch.zeros((1, x_size[0] // 2, x_size[1] // 2, 1))
        h_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size, -self.shift_size // 2), slice(-self.shift_size // 2, None))
        w_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size, -self.shift_size // 2), slice(-self.shift_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                permuted_window_mask[:, h, w, :] = cnt
                cnt += 1
        permuted_windows = window_partition(permuted_window_mask, self.permuted_window_size)
        permuted_windows = permuted_windows.view(-1, self.permuted_window_size * self.permuted_window_size)
        
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.after_norm1(x)
        x = x.view(b, h, w, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)
        x = self.after_attention(x)
        x = shortcut + self.drop_path(x)
        x = self.residual_after_attention(x)
        x = self.residual_after_mlp(x + self.drop_path(self.after_mlp(self.mlp(self.after_norm2(self.norm2(x)), x_size))))
        return x
    
    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * 25
        flops += self.dim * h * w
        return flops

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            PSA_Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.norm2(x)
        x = x.transpose(1, 2).view(x.shape[0], self.dim, x_size[0], x_size[1])
        return x
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

class Head(nn.Module):
    def __init__(self, embed_dim=24, drop_path=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(1, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, padding=1)
        )
        alpha_0 = 1e-2
        self.alpha = nn.Parameter(
            alpha_0 * torch.ones((1, self.embed_dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.alpha * self.block(x))
        return x

class Tail(nn.Module):
    def __init__(self, embed_dim=24):
        super().__init__()
        self.embed_dim = embed_dim
        self.block = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim // 2, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.block(x)

class RB(nn.Module):
    def __init__(self, channel, relu_slope, use_HIN=False):
        super(RB, self).__init__()
        self.use_HIN = use_HIN
        if use_HIN:
            self.norm = nn.InstanceNorm2d(channel // 2, affine=True)
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out)) + x
        return out

class Unet_Enc(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Unet_Enc, self).__init__()
        self.conv_forward = Head(embed_dim=dim_ladder[0])
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim_ladder[0], dim_ladder[1] // 4, 3, padding=1, bias=False),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.down2 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[2] // 4, 3, padding=1, bias=False),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.RB3 = RB(dim_ladder[2], 0.2, True)

    def forward(self, x_in):
        x_level_32 = self.RB1(self.enhance1(self.conv_forward(x_in)))
        x_level_64 = self.RB2(self.enhance2(self.down1(x_level_32)))
        x_level_128 = self.RB3(self.enhance3(self.down2(x_level_64)))
        x_features = [x_level_32, x_level_64, x_level_128]
        return x_features

class Unet_Dec(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Unet_Dec, self).__init__()
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.conv_backward = Tail(embed_dim=dim_ladder[0])

    def forward(self, x_features):
        x_level_64 = self.merge2(torch.cat([self.up2(self.enhance3(self.RB3(x_features[2]))), x_features[1]], dim=1))
        x_level_32 = self.merge1(torch.cat([self.up1(self.enhance2(self.RB2(x_level_64))), x_features[0]], dim=1))
        x_opt = self.conv_backward(self.enhance1(self.RB1(x_level_32)))
        return x_opt

class Feature_Dec(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Feature_Dec,self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.1]))
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.enhance3 = TransformerBlock(dim_ladder[2], 8, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.enhance2 = TransformerBlock(dim_ladder[1], 4, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=False),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.enhance1 = TransformerBlock(dim_ladder[0], 2, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.conv_backward = Tail(embed_dim=dim_ladder[0])

    def forward(self, x_features):
        x_level_64 = self.merge2(torch.cat([self.up2(self.enhance3(self.RB3(x_features[2]))), x_features[1]], dim=1))
        x_level_32 = self.merge1(torch.cat([self.up1(self.enhance2(self.RB2(x_level_64))), x_features[0]], dim=1))
        x_opt = self.alpha * self.conv_backward(self.enhance1(self.RB1(x_level_32)))
        return x_opt

class Mid_Dec(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Mid_Dec, self).__init__()
        self.RB3 = RB(dim_ladder[2], 0.2, True)
        self.up2 = nn.Sequential(
            nn.Conv2d(dim_ladder[2], dim_ladder[1] * 4, 3, padding=1, bias=bias),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(dim_ladder[1]*2, dim_ladder[1], 1, 1, 0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.up1 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[0] * 4, 3, padding=1, bias=bias),
            Rearrange("b (c t1 t2) h w -> b c (h t1) (w t2)", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(dim_ladder[0]*2, dim_ladder[0], 1, 1, 0)
        self.RB1 = RB(dim_ladder[0], 0.2, True)

    def forward(self, x_features):
        x_level_128 = self.RB3(x_features[2])
        x_level_64 = self.RB2(self.merge2(torch.cat([self.up2(x_level_128), x_features[1]], dim=1))) + x_features[1]
        x_level_32 = self.RB1(self.merge1(torch.cat([self.up1(x_level_64), x_features[0]], dim=1))) + x_features[0]
        return [x_level_32,x_level_64,x_level_128]

class Mid_Enc(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Mid_Enc,self).__init__()
        self.RB1 = RB(dim_ladder[0], 0.2, True)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim_ladder[0], dim_ladder[1] // 4, 3, padding=1, bias=bias),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.merge1 = nn.Conv2d(2*dim_ladder[1],dim_ladder[1],1,1,0)
        self.RB2 = RB(dim_ladder[1], 0.2, True)
        self.down2 = nn.Sequential(
            nn.Conv2d(dim_ladder[1], dim_ladder[2] // 4, 3, padding=1, bias=bias),
            Rearrange("b c (h t1) (w t2) -> b (c t1 t2) h w", t1=2, t2=2),
        )
        self.merge2 = nn.Conv2d(2*dim_ladder[2],dim_ladder[2],1,1,0)
        self.RB3 = RB(dim_ladder[2], 0.2, True)

    def forward(self, x_features):
        x_level_32 = self.RB1(x_features[0])
        x_level_64 = self.RB2(self.merge1(torch.cat([self.down1(x_level_32),x_features[1]],dim=1))) + x_features[1]
        x_level_128 = self.RB3(self.merge2(torch.cat([self.down2(x_level_64),x_features[2]],dim=1))) + x_features[2]
        return [x_level_32, x_level_64, x_level_128]

class Mid_Fusion(nn.Module):
    def __init__(self, dim_ladder, bias=False):
        super(Mid_Fusion,self).__init__()
        self.mid_dec = Mid_Dec(dim_ladder=dim_ladder,bias=bias)
        self.mid_enc = Mid_Enc(dim_ladder=dim_ladder,bias=bias)

    def forward(self,x_features):
        x_features = self.mid_enc(self.mid_dec(x_features))
        return x_features

class SWin_Stage(nn.Module):
    def __init__(self, dim, resolution, depth, bias=False):
        super(SWin_Stage, self).__init__()
        self.dim = dim
        self.swin_forward = BasicLayer(dim, (resolution, resolution), depth, 4, 8)
        self.catt_forward = TransformerBlock(dim, dim // 16, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.mid_merge = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1)
        )
        self.soft_thr = nn.Parameter(torch.full((2 * dim, 1, 1), 0.005))
        self.swin_backward = BasicLayer(dim, (resolution, resolution), depth, 4, 8)
        self.catt_backward = TransformerBlock(dim, dim // 16, 2.66, bias=bias, LayerNorm_type="WithBias")
        self.merge = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x_feature):
        x_in_swin = self.swin_forward(x_feature)
        x_in_catt = self.catt_forward(x_feature)
        x_in = self.mid_merge(torch.cat([x_in_swin, x_in_catt], dim=1))
        x_out = torch.mul(torch.sign(x_in), F.relu(torch.abs(x_in) - self.soft_thr))
        x_out_swin, x_out_catt = torch.split(x_out, split_size_or_sections=self.dim, dim=1)
        x_opt_1 = self.swin_backward(x_out_swin) + x_feature
        x_opt_2 = self.catt_backward(x_out_catt) + x_feature
        x_opt = self.merge(torch.cat([x_opt_1, x_opt_2], dim=1))
        return x_opt

class Denoise_Block(nn.Module):
    def __init__(self, dim_ladder, resolution, bias=False):
        super(Denoise_Block, self).__init__()
        self.r = [1, 2, 4]
        self.nf = [dim_ladder[0], dim_ladder[1], dim_ladder[2]]
        
        self.Grad1 = nn.Sequential(
            nn.Conv2d(self.r[0] ** 2 + 2*self.nf[0], self.nf[0], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[0],self.nf[0],3,padding=1),
            TransformerBlock(self.nf[0], self.nf[0] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr1 = SWin_Stage(self.nf[0], resolution, 2)

        self.Grad2 = nn.Sequential(
            nn.Conv2d(self.r[1] ** 2 + 2*self.nf[1], self.nf[1], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[1], self.nf[1], 3, padding=1),
            TransformerBlock(self.nf[1], self.nf[1] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr2 = SWin_Stage(self.nf[1], resolution // 2, 2)

        self.Grad3 = nn.Sequential(
            nn.Conv2d(self.r[2] ** 2 + 2*self.nf[2], self.nf[2], 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.nf[2], self.nf[2], 3, padding=1),
            TransformerBlock(self.nf[2], self.nf[2] // 16, 2.66, bias=bias, LayerNorm_type="WithBias"),
            nn.Sigmoid()
        )
        self.swin_thr3 = SWin_Stage(self.nf[2], resolution // 4, 2)

    def forward(self, x_features, Phiweight, PhiTb):
        x_1 = F.pixel_shuffle(x_features[0], upscale_factor=self.r[0])
        b, c, h, w = x_1.shape
        PhiT_Phi_x = PhiTPhi_fun(x_1.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[0])
        grad_1 = self.Grad1(
            torch.cat([x_features[0], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[0])], dim=1))
        x_features[0] = x_features[0] - grad_1
        x_features[0] = self.swin_thr1(x_features[0])

        x_2 = F.pixel_shuffle(x_features[1], upscale_factor=self.r[1])
        b, c, h, w = x_2.shape
        PhiT_Phi_x = PhiTPhi_fun(x_2.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[1])
        grad_2 = self.Grad2(
            torch.cat([x_features[1], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[1])], dim=1))
        x_features[1] = x_features[1] - grad_2
        x_features[1] = self.swin_thr2(x_features[1])

        x_3 = F.pixel_shuffle(x_features[2], upscale_factor=self.r[2])
        b, c, h, w = x_3.shape
        PhiT_Phi_x = PhiTPhi_fun(x_3.reshape(-1, 1, h, w), Phiweight).reshape(b, c, h, w)
        PhiT_Phi_x = F.pixel_unshuffle(PhiT_Phi_x, downscale_factor=self.r[2])
        grad_3 = self.Grad3(
            torch.cat([x_features[2], PhiT_Phi_x, F.pixel_unshuffle(PhiTb, self.r[2])], dim=1))
        x_features[2] = x_features[2] - grad_3
        x_features[2] = self.swin_thr3(x_features[2])

        return x_features

class AUV_Net(nn.Module):
    def __init__(self, layer_num=7, resolution=64, rate=10):
        super(AUV_Net, self).__init__()

        self.patch_size = resolution
        self.n_input = int(rate * 0.01 * (self.patch_size ** 2))
        self.layer_num = layer_num

        self.Phiweight = nn.Parameter(
            init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))

        dim_ladder = [48, 64, 80]

        self.Encoder = Unet_Enc(dim_ladder=dim_ladder)

        block_list = []
        for i in range(0, self.layer_num - 1):
            block_list.append(Denoise_Block(dim_ladder=dim_ladder, resolution=resolution))
            block_list.append(Mid_Fusion(dim_ladder=dim_ladder))
        block_list.append(Denoise_Block(dim_ladder=dim_ladder, resolution=resolution))
        self.denoise_stage = nn.ModuleList(block_list)

        self.Decoder = Unet_Dec(dim_ladder=dim_ladder)
        self.Features_Dec = Feature_Dec(dim_ladder=dim_ladder)

    def forward(self, input):
        Phix = F.conv2d(input, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        PhiTb = F.conv_transpose2d(Phix, self.Phiweight, stride=self.patch_size)

        x = PhiTb
        x_features = self.Encoder(x)
        
        shotcut = [torch.zeros_like(x_features[i]) for i in range(len(x_features))]

        for i in range(0, self.layer_num - 1):
            x_features = self.denoise_stage[2*i](x_features, self.Phiweight, PhiTb)
            for j in range(len(shotcut)):
                shotcut[j] = shotcut[j] + x_features[j]
            x_features = self.denoise_stage[2*i+1](x_features)
        
        x_features = self.denoise_stage[2*(self.layer_num-1)](x_features, self.Phiweight, PhiTb)
        for j in range(len(shotcut)):
            shotcut[j] = shotcut[j] + x_features[j]

        x_opt = self.Decoder(x_features) + self.Features_Dec(shotcut)
        return x_opt
