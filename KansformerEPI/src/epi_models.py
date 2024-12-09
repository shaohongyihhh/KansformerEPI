#!/usr/bin/env python3

import argparse, os, sys, time
#import warnings, json, gzip

import torch
import torch.nn as nn  # torch.nn 是核心神经网络库，包含构建神经网络的基础组件
import torch.nn.functional as F  # torch.nn.functional 提供了一些常用的函数，如激活函数relu
from torch.autograd import Variable
import numpy as np

# from performer_pytorch import SelfAttention

from typing import Dict, List
from kanformer import KANTransformer
from kan import KAN, KANLinear


# 继承自 nn.Module，这是所有神经网络模块的基类
# 实现了一个位置相关的前馈神经网络层
# 这种结构在 Transformer 模型中非常常见有助于增强模型的表达能力和稳定性
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise(位置相关的线性变换)  d_in:输入的维度  d_hid:隐藏层的维度
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)  # 层归一化，用于规范化输入，以提高模型稳定性
        self.dropout = nn.Dropout(dropout)    # Dropout 层，用于防止过拟合

    # 前向传播方法
    def forward(self, x):

        residual = x   # x 是输入张量，形状为 [batch_size, seq_len, d_in] 
                       # 残差连接，用于保存输入以用于后面的残差连接

        x = self.w_2(F.relu(self.w_1(x)))   # 第一层线性变换和ReLu激活
        x = self.dropout(x)    # 应用dropout，随机丢弃一些神经元的输出
        x += residual    # 残差连接:将 Dropout 的输出与原始输入相加

        x = self.layer_norm(x)   # 对结果应用层归一化

        return x


# 继承自nn.Module 是神经网络模块的基类
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):   # d_hid:隐藏维度,表示每个位置的编码向量的维度，n_position:位置的数量，默认值200
        super(PositionalEncoding, self).__init__()

        # Not a parameter(参数)
        # self.register_buffer:将pos_table 注册为模块的持久缓冲区,但不是可训练的参数,训练过程不会被优化
        # self._get_sinusoid_encoding_table:调用内部方法生成正弦位置编码表
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    # 生成正弦位置编码表的方法
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''  # 正弦位置编码表
        # TODO: make it with torch instead of numpy

        # 内部函数
        # 输入position：位置索引   
        # 输出:一个长度为d_hid 的向量，计算每个维度上的位置角度
        def get_position_angle_vec(position):    
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # 生成编码表 sinusoid_table
        # 创建一个大小为 [n_position, d_hid] 的数组，每一行是一个位置的编码向量
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i   偶数维度使用sin
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 技术维度使用cos，以确保不同维度的编码具有不同的周期性特征

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)   # 转换为 PyTorch 的 FloatTensor 类型，并增加一个维度，使其形状为 [1, n_position, d_hid]

    def forward(self, x):
        # self.pos_table[:, :x.size(1)]：取位置编码表的前 seq_len 个位置编码;.clone().detach():确保位置编码表在计算图中被分离，以防止其被意外修改
        return x + self.pos_table[:, :x.size(1)].clone().detach()  # 输出:将输入张量 x 与位置编码表相加


class KansformerEPI(nn.Module):
    def __init__(self, in_dim: int, 
            cnn_channels: List[int], cnn_sizes: List[int], cnn_pool: List[int],
            enc_layers: int, num_heads: int, d_inner: int,
            da: int, r: int, att_C: float,
            fc: List[int], fc_dropout: float, seq_len: int=-1, pos_enc: bool=False,
            **kwargs):
        super(KansformerEPI, self).__init__()
        
        major, minor = torch.__version__.split('.')[:2]
        assert int(major) >= 1 and int(minor) >= 6, "PyTorch={}, while PyTorch>=1.6 is required".format(torch.__version__)
        if int(minor) < 9:
            self.transpose = True
        else:
            self.transpose = False

        if pos_enc:
            assert seq_len > 0
        
        self.cnn = nn.ModuleList()
        self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_dim, 
                        out_channels=cnn_channels[0], 
                        kernel_size=cnn_sizes[0], 
                        padding=cnn_sizes[0] // 2),
                    nn.BatchNorm1d(cnn_channels[0]),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(cnn_pool[0])
                )
            )
        seq_len //= cnn_pool[0]
        for i in range(len(cnn_sizes) - 1):
            self.cnn.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=cnn_channels[i], 
                            out_channels=cnn_channels[i + 1], 
                            kernel_size=cnn_sizes[i + 1],
                            padding=cnn_sizes[i + 1] // 2),
                        nn.BatchNorm1d(cnn_channels[i + 1]),
                        nn.LeakyReLU(),
                        nn.MaxPool1d(cnn_pool[i + 1])
                )
            )
            seq_len //= cnn_pool[i + 1]

        self.lstm = nn.LSTM(180, 90, 2, bidirectional=True)

        if pos_enc:
            self.pos_enc = PositionalEncoding(d_hid=cnn_channels[-1], n_position=seq_len)
        else:
            self.pos_enc = None
        
        self.encoder = KANTransformer(embed_dim=cnn_channels[-1], depth=enc_layers,
                    num_heads=num_heads)

        self.da = da
        self.r = r
        self.att_C = att_C
        self.att_first = nn.Linear(cnn_channels[-1], da)
        # self.att_first = KANLinear(in_features=cnn_channels[-1], out_features=da, base_activation=torch.nn.Tanh)
        self.att_first.bias.data.fill_(0)
        self.att_second = nn.Linear(da, r)
        # self.att_second = KANLinear(in_features=da, out_features=r, base_activation=torch.nn.SiLU)
        self.att_second.bias.data.fill_(0)


        if fc[-1] != 1:
            fc.append(1)
        self.fc = nn.ModuleList()
        self.fc.append(
                nn.Sequential(
                    nn.Dropout(p=fc_dropout),
                    nn.Linear(cnn_channels[-1] * 4, fc[0])
                )
            )

        for i in range(len(fc) - 1):
            self.fc.append(
                    nn.Sequential(
                        # nn.ReLU(),
                        # nn.Linear(fc[i], fc[i + 1])
                        KANLinear(in_features=fc[i], out_features=fc[i+1], base_activation=torch.nn.SiLU)
                    )
                )
        self.fc.append(nn.Sigmoid()) #1/(1+e^-x)
        self.fc_dist = nn.Sequential(
                    # nn.Linear(cnn_channels[-1] * 4, cnn_channels[-1]),
                    # nn.ReLU(),
                    # nn.Linear(cnn_channels[-1], 1),
                    KANLinear(in_features=cnn_channels[-1] * 4, out_features=cnn_channels[-1]),
                    KANLinear(in_features=cnn_channels[-1], out_features=1),
                )

 



    def forward(self, feats, enh_idx, prom_idx, return_att=False):
        # feats: (B, D, S)
        if type(feats) is tuple:
            feats, length = feats
        else:
            length = None
        div = 1
        for cnn in  self.cnn:
            div *= cnn[-1].kernel_size
            enh_idx = torch.div(enh_idx, cnn[-1].kernel_size, rounding_mode="trunc")
            prom_idx = torch.div(prom_idx, cnn[-1].kernel_size, rounding_mode="trunc")
            feats = cnn(feats)
        feats = feats.transpose(1, 2) # -> (B, S, D)

        # 加入LSTM
        feats = feats.transpose(0, 1) # -> (S, B, D)
        feats=self.lstm(feats)
        feats, _ = feats
        feats = feats.transpose(0, 1) # -> (B, S, D)

        batch_size, seq_len, feat_dim = feats.size()
        if self.pos_enc is not None:
            feats = self.pos_enc(feats)
        if self.transpose:
            feats = feats.transpose(0, 1)
        feats = self.encoder(feats) # (B, S, D)
        if self.transpose:
            feats = feats.transpose(0, 1)

        # print(f'before att_first: feats.shape{feats.shape}')
        out = torch.tanh(self.att_first(feats)) # (B, S, da)
        # print(f'after att_first: feats.shape{feats.shape}')
        # out = self.att_first(feats)
        if length is not None:
            length = torch.div(length, div, rounding_mode="trunc")
            max_len = max(length)
            mask = torch.cat((
                [torch.cat((torch.ones(1, m, self.da), torch.zeros(1, max_len - m, self.da)), dim=1) for m in length]
            ), dim=0)
            assert mask.size() == out.size()
            out = out * mask.to(out.device)
            del mask
        out = F.softmax(self.att_second(out), 1) # (B, S, r)
        att = out.transpose(1, 2) # (B, r, S)
        del out
        seq_embed = torch.matmul(att, feats) # (B, r, D)
        # print(seq_embed.size())
        base_idx = seq_len * torch.arange(batch_size) # .to(feats.device)
        enh_idx = enh_idx.long().view(batch_size) + base_idx
        prom_idx = prom_idx.long().view(batch_size) + base_idx
        feats = feats.reshape(-1, feat_dim)
        seq_embed = torch.cat((
            feats[enh_idx, :].view(batch_size, -1), 
            feats[prom_idx, :].view(batch_size, -1),
            seq_embed.mean(dim=1).view(batch_size, -1),
            seq_embed.max(dim=1)[0].view(batch_size, -1)
        ), axis=1)
        del feats
        # feats = torch.cat((feats.max(dim=1)[0].squeeze(1), feats.mean(dim=1).squeeze(1)), dim=1)
        dists = self.fc_dist(seq_embed)

        
        for fc in self.fc:
            seq_embed = fc(seq_embed)

        if return_att:
            return seq_embed, dists, att
        else:
            del att
            return seq_embed

    def l2_matrix_norm(self, m):                                                                                        
        return torch.sum(torch.sum(torch.sum(m**2, 1), 1)**0.5).type(torch.cuda.DoubleTensor)



def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()
    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

