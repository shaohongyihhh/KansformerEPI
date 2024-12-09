import torch.nn as nn
import torch
from kan import KAN, KANLinear
import torch.nn.functional as F 
from functools import partial 
from collections import OrderedDict
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    在每个样本上应用 Stochastic Depth（随机深度）来丢弃路径（当应用于残差块的主路径时）。
    这与EfficientNet等网络创建的 DropConnect 实现相同，但是，原始名称有误导性，因为'Drop Connect' 是另一篇论文中不同形式的 dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择了
    将层和参数名称更改为 'drop path'，而不是将 DropConnect 作为层名称并使用 'survival rate' 作为参数。
    """
    if drop_prob == 0. or not training:  # 如果 drop_prob 为 0 或者模型不处于训练模式，直接返回输入张量 x，不进行任何操作
        return x
    keep_prob = 1 - drop_prob  # 计算保持路径的概率，即 1 减去丢弃路径的概率
    # 创建一个与输入张量 x 的形状兼容的 shape
    # (x.shape[0],) 表示保持批次维度，(1,) * (x.ndim - 1) 表示在其他维度上保持单一值
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 创建一个与输入张量形状兼容的随机张量 random_tensor
    # 这个张量的值在 keep_prob 和 1 之间（包含 keep_prob）
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 使用 floor_() 方法将随机张量中的值二值化
    # 所有大于等于 1 的值将变为 1，所有小于 1 的值将变为 0
    random_tensor.floor_()
    # 将输入张量 x 按照 keep_prob 进行缩放（除以 keep_prob）
    # 然后与二值化的 random_tensor 相乘，以实现按比例丢弃路径
    output = x.div(keep_prob) * random_tensor
    return output  # 返回处理后的输出张量

class DropPath(nn.Module):
    """
    用于每个样本的Drop paths（随机深度）（当应用于残差块的主路径时）。
    """

    def __init__(self, drop_prob=None):
        """
        初始化 DropPath 类。

        Args:
            drop_prob: 丢弃路径的概率，默认为 None。
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量。

        Returns:
            经过 DropPath 操作后的张量。
        """
        # 调用 drop_path 函数，传入输入张量 x、丢弃路径概率 self.drop_prob 和训练模式 self.training
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,  # 注意力头的数量，默认为8
                 qkv_bias=False,  # 是否为注意力查询、键、值添加偏置，默认为False
                 qk_scale=None,  # 查询和键的缩放因子，默认为 None
                 attn_drop_ratio=0.,  # 注意力权重的丢弃率，默认为0
                 proj_drop_ratio=0.):  # 输出投影的丢弃率，默认为0
        super(Attention, self).__init__()
        # 初始化注意力层 
        self.num_heads = num_heads  # 设置注意力头的数量
        head_dim = dim // num_heads  # 计算每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 设置缩放因子，若未提供则默认为头维度的倒数的平方根
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 初始化注意力机制中的查询、键、值的线性变换
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 初始化用于丢弃注意力权重的Dropout层
        self.proj = nn.Linear(dim, dim)  # 初始化输出投影的线性变换
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 初始化用于丢弃输出投影的Dropout层

    def forward(self, x):
        # 获取输入张量 x 的形状信息
        # x: [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # 对输入张量 x 进行线性变换得到查询、键、值
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 将查询、键、值分离出来
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 计算注意力权重
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 对输出进行线性变换
        x = self.proj_drop(x)  # 对输出进行 Dropout 操作
        return x



class Block(nn.Module):
    """
    在KANTransformer中使用的基本块
    """
    """
    初始化Block类。

    Args:
        dim: 输入维度。
        num_heads: 注意力头的数量。
        mlp_ratio: MLP隐藏层维度与输入维度的比率，默认为4。
        qkv_bias: 是否为注意力查询、键、值添加偏置，默认为False。
        qk_scale: 查询和键的缩放因子，默认为None。
        drop_ratio: 通用的丢弃率，默认为0。
        attn_drop_ratio: 注意力权重的丢弃率，默认为0。
        drop_path_ratio: DropPath操作的丢弃率，默认为0。
        act_layer: 激活函数，默认为GELU激活函数。
        norm_layer: 规范化层，默认为LayerNorm。
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # 初始化第一个规范化层和注意力机制
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 初始化DropPath操作
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # 初始化第二个规范化层和KAN
        self.norm2 = norm_layer(dim)
        self.kan = KAN([dim, 64, dim])
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量。

        Returns:
            经过Block操作后的张量。
        """
        b, t, d = x.shape  # 获取输入张量 x 的批次大小、序列长度和特征维度
        # 对输入张量进行规范化、注意力机制、DropPath操作
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 对输入张量进行规范化、KAN操作、DropPath操作
        # 对输入张量进行规范化，然后将其形状重塑为 [b*t, d]，其中 b 为批次大小，t 为序列长度，d 为特征维度
        # 将重塑后的张量再次重塑为 [b, t, d] 的形状，恢复原始的批次大小、序列长度和特征维度
        # x = x + self.drop_path(self.kan(self.norm2(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))
        return x


# 新加：全局-局部注意力编码器
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# 新加：全局-局部注意力编码器
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# 新加：全局-局部注意力编码器
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.act_fn1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.act_fn2 = h_sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.act_fn1(y)
        y = self.fc2(y)
        y = self.act_fn2(y)
        y = y.view(b, c, 1, 1)
        return x * y

# 新加：全局-局部注意力编码器
class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 新加：全局-局部注意力编码器
class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        self.layers = nn.ModuleList([])
        # the first linear layer is replaced by 1x1 convolution.
        self.layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                self.layers = dp + self.layers
            else:
                self.layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                self.layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                self.layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        self.layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])

    def forward(self, x):
        h = x
        for layer in self.layers:
            x = layer(x)
        x = x + h
        return x


class BlockWithCNN(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super(BlockWithCNN, self).__init__()
        self.hidden_size = dim
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        # self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.conv = LocalityFeedForward(self.hidden_size, self.hidden_size, 1, 4, act="hs+se")
        # self.ffn = Mlp(config)
        self.attn = Attention(self.hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        # x, weights = self.attn(x)
        x = self.attn(x)
        x = x + h

        batch_size, num_token, embed_dim = x.shape  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token - 1))
        # Split the class token and the image token.
        cls_token, x = torch.split(x, [1, num_token - 1], dim=1)  # (B, 1, dim), (B, 196, dim)
        # Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)  # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)  # (B, 196, dim)
        # Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)

        # return x, weights
        return x



class KANTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.1,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        """
        初始化 Vision Transformer 模型。

        Args:
            img_size (int, tuple): 输入图像大小。
            patch_size (int, tuple): patch 大小。
            in_c (int): 输入通道数。
            num_classes (int): 分类头中的类别数。
            embed_dim (int): 嵌入维度。
            depth (int): Transformer 的深度。
            num_heads (int): 注意力头的数量。
            mlp_ratio (int): MLP 隐藏维度与嵌入维度的比率。
            qkv_bias (bool): 是否为注意力查询、键、值添加偏置。
            qk_scale (float): 查询和键的缩放因子。
            representation_size (Optional[int]): 如果设置，则启用并设置表示层（预对数层）的大小。
            distilled (bool): 模型是否包含蒸馏标记和头部，如 DeiT 模型。
            drop_ratio (float): dropout 比率。
            attn_drop_ratio (float): 注意力 dropout 比率。
            drop_path_ratio (float): 随机深度率。
            embed_layer (nn.Module): patch 嵌入层。
            norm_layer: (nn.Module): 规范化层。
            act_layer: 激活函数。
        """
        super(KANTransformer, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-5)
        act_layer = act_layer or nn.GELU
       
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 初始化随机深度率
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 随机深度衰减规则
        # 初始化 Transformer 块
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)


        # 新加：全局-局部注意力编码器
        self.cnn_block = BlockWithCNN(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)

    def forward(self, x):
        """
        完整的前向传播。

        Args:
            x (tensor): 输入张量。

        Returns:
            tensor: 输出张量。
        """
        x = self.blocks(x)  # 经过 Transformer 块处理特征张量
        x = self.norm(x)  # 对处理后的特征张量进行规范化
        return x
