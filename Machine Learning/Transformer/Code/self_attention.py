import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductionAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2) #沿哪一维实施softmax

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u/self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.bmm(attn, v)
        return attn, output
    

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 用于投影变换mlp
        # nn.Linear(input, output)
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductionAttention(scale=np.power(d_k, 0.5))
        self.fc_concatOutput = nn.Linear(
            n_head * d_v, d_o)  # concat -> mlp -> output

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        # 投影变化，单头变多头
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        # view method1: (batch, n_q, n_head * d_q) -> (batch, n_q, n_head, d_q)
        # permute method: 将tensor维度重排列为 (n_head, batch, n_q, d_q)
        # contiguous method: 确保张量在内存中是连续存储的
        # view method2: (n_head, batch, n_q, d_q) -> (n_head * batch, n_q, d_q)
        q = q.view(batch, n_q, n_head, d_q).permute(
            2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(
            2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(
            2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            # repeat(n_head, 1, 1): 将mask沿第0维复制 n_head次，其他维度不变
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)  # 当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(
            1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # Concat
        output = self.fc_concatOutput(output)  # 投影变换得到最终输出
        return attn, output
    

class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(
            n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 2, 2 #个数
    d_q, d_k, d_v = 64, 64, 64 #维度
    batch = 1

    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attention = ScaledDotProductionAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)
    print(attn)
    print(output)


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    batch = 32

    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)
    mask = torch.zeros(batch, n_q, n_k).bool()

    mha = MultiHeadAttention(
        n_head=8, d_k_=128, d_v_=64, d_k=256, d_v=128, d_o=128)
    attn, output = mha(q, k, v, mask=mask)
    print(attn)
    print(output)
    print(attn.size())
    print(output.size())


if __name__ == "__main__":
    n_x = 4
    d_x = 80
    batch = 2

    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()

    selfattn = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=80, d_o=80)
    attn, output = selfattn(x, mask=mask)

    print(attn.size())
    print(output.size())
