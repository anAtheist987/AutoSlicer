import torch
from torch import nn

from typing import Sequence, Union, Optional

from .model import Slicer, activation, normalization, zero_module


class GEGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, mult=4, dropout=0.):
        super().__init__()
        inner_channels = int(in_channels * mult)

        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            GEGLU(in_channels, inner_channels),
            nn.Dropout(dropout),
            zero_module(nn.Linear(inner_channels, in_channels)),
        )

    def forward(self, x):
        return self.net(x) + x


class CrossAttn(nn.Module):
    """
    Multi Head Cross Attention or Self Attention
    """
    def __init__(
            self,
            in_channels, context_channels=None,
            head_num=8, head_channels=64,
            dropout=0.,
    ):
        super().__init__()
        if context_channels is None:
            context_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.context_channels = context_channels

        self.head_num = head_num
        self.head_channels = head_channels
        inner_channels = head_num * head_channels

        self.norm0 = nn.LayerNorm(in_channels)
        self.q = nn.Linear(in_channels, inner_channels)
        self.k = nn.Linear(context_channels, inner_channels)
        self.v = nn.Linear(context_channels, inner_channels)

        self.proj_out = nn.Sequential(
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(inner_channels, in_channels))
            if inner_channels != in_channels else
            nn.Identity(),
        )

    def forward(self, x, context=None):
        """
        cross attention with context
        perform self attention if context is None
        :param x: [N, L0, C] or [N*h*w, W*W, C]
        :param context: Optional([N, L1, context_channels] or [N*h*w, W*W, C])
        :return: [N, L0, C]
        """
        if context is None:
            context = x
        assert context.shape[-1] == self.context_channels, "unexpected context channels."

        shortcut = x

        x = self.norm0(x)
        q = self.q(x)  # n,l0,h*d
        k = self.k(context)  # n,l1,h*d
        v = self.v(context)  # n,l1,h*d

        q = q.unflatten(-1, (self.head_num, self.head_channels)).transpose(1, 2)   # n,h,l0,d
        v = v.unflatten(-1, (self.head_num, self.head_channels)).transpose(1, 2)   # n,h,l1,d
        k = k.unflatten(-1, (self.head_num, self.head_channels)).permute(0, 2, 3, 1)   # n,h,d,l1

        attn = q @ k  # n,h,l0,l1
        attn = attn * (self.head_channels ** (-0.5))

        attn = torch.nn.functional.softmax(attn, dim=-1)
        # attend to values
        x = attn @ v  # n,h,l0,c1

        x = (
            x.transpose(1, 2)  # n,l0,h,d
            .flatten(-2, -1)  # n,l0,h*d
        )

        x = self.proj_out(x)  # n,l0,c
        return x + shortcut


class SelfAttn(nn.Module):
    def __init__(
            self,
            in_channels,
            head_num=8, head_channels=64,
            dropout=0.,
    ):
        super().__init__()
        self.attn = CrossAttn(
            in_channels=in_channels, context_channels=in_channels,
            head_num=head_num, head_channels=head_channels,
            dropout=dropout,
        )

    def forward(self, x):
        """
        perform self attention
        :param x: [N, L0, C] or [N*h*w, W*W, C]
        :return: [N, L0, C]
        """
        return self.attn(x, x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            context_channels: Optional[int] = None,
            head_num=8, head_channels=64,
            dropout=0.,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.context_channels = context_channels

        inner_channels = head_num * head_channels

        self.cross_attn = None
        if context_channels is not None:
            self.cross_attn = CrossAttn(
                    in_channels=inner_channels, context_channels=context_channels,
                    head_num=head_num, head_channels=head_channels,
                    dropout=dropout,
                )

        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()
        if in_channels != inner_channels:
            self.proj_in = nn.Linear(in_channels, inner_channels)
            self.proj_out = nn.Linear(inner_channels, in_channels)

        self.self_attn = SelfAttn(
                in_channels=inner_channels,
                head_num=head_num, head_channels=head_channels,
                dropout=dropout,
            )
        self.ff = FeedForward(
                in_channels=inner_channels, dropout=dropout
            )

    def forward(self, x, context=None):
        """
        :param x: [N, L0, C]
        :param context: [N, L1, context_channels]
        :return: [N, L0, C]
        """
        x = self.proj_in(x)
        x = self.self_attn(x)
        if self.cross_attn is not None:
            x = self.self_attn(x, context)
        x = self.ff(x)
        x = self.proj_out(x)
        return x


class Quantizer(nn.Module):
    def __init__(
            self, in_channel: int, out_channel: int, group=2, entry=320,
    ):
        super(Quantizer, self).__init__()
        assert out_channel % group == 0
        self.codebooks: nn.ModuleList[nn.Embedding] = nn.ModuleList(
            [nn.Embedding(entry, out_channel // group) for _ in range(group)]
        )
        self.weight_proj = nn.Linear(in_channel, group * entry)
        self.proj_q = nn.Linear(out_channel, out_channel)

        self.group = group
        self.entry = entry

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: [..., C]

        Returns
        -------

        """
        x = self.weight_proj(x)  # [..., entry*group]
        x = torch.stack(x.chunk(self.group, -1), dim=0)  # [group, ..., entry]
        x = nn.functional.gumbel_softmax(x, 1., hard=False, dim=-1)

        # sigma(group, exp(-sigma(entry, p*log(g)))) and mean on batch
        prob_perplexity = (-x * torch.log(x)).sum(-1).exp().sum(0).mean()
        loss = (1 - prob_perplexity / (self.group * self.entry))

        code_part = [
            part @ codebook.weight
            for part, codebook in zip(x, self.codebooks)
        ]  # list[..., Cout//group]
        x = torch.cat(code_part, -1)  # [..., Cout]
        return x, loss


def make_mask(batch: int, length: int, p=0.2, l=2) -> torch.BoolTensor:
    """

    Parameters
    ----------
    batch
    length: length of mask
    p: chance of being masked
    l: length of a mask point if it is being chosen

    Returns
    -------
    mask: torch.BoolTensor [N, L]
    """
    mask = (torch.rand(batch, length) < p).to(dtype=torch.float32)
    mask = torch.constant_pad_nd(mask, (0, l - 1), value=0)
    mask = torch.conv1d(mask[:, None, :], torch.ones(1, 1, l)) > 0
    return mask.squeeze(1)


class ContextPredictor(Slicer):
    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),
            groups: Union[int, Sequence[int]] = 4,
            res_num: Union[int, Sequence[int]] = 1,
            down_sample_scale: Union[int, Sequence[int]] = 3,
            context_channel=192, context_num_layers=4, context_num_heads=4,
            codebook_group=2, codebook_entry=320,
            out_ch=128,
    ):
        super(ContextPredictor, self).__init__(
            channels=channels, groups=groups, res_num=res_num, down_sample_scale=down_sample_scale,
            context_channel=context_channel, context_num_layers=0, out_ch=out_ch,
        )

        self.masked_code = nn.Parameter(torch.randn(context_channel))
        self.quantizer = Quantizer(channels[-1], out_channel=out_ch, group=codebook_group, entry=codebook_entry)

        self.pre_extract = nn.Linear(channels[-1], context_channel)
        self.pos_emb = nn.Conv1d(context_channel, context_channel, kernel_size=65, padding="same", groups=8)

        self.context_model = nn.Sequential()
        self.context_model.add_module("act0", activation())
        for i in range(context_num_layers):
            self.context_model.add_module(f"tranformer{i}", TransformerBlock(
                context_channel, head_num=context_num_heads, head_channels=context_channel // context_num_heads,
                dropout=0.2,
            ))
        self.context_model.add_module("norm_fin", nn.LayerNorm(context_channel))
        self.context_model.add_module("proj_fin", nn.Linear(context_channel, out_ch))

        # for logging
        self.l_contrastive = None
        self.l_diversity = None
        self.l_penalty = None

    def forward(self, x, _=None):
        latent = self.encoder(x)  # [N, L] -> [N, C, Lout]
        latent = latent.transpose(1, 2)  # [N, C, L] -> [N, L, C]
        mask = make_mask(*latent.shape[0:2], p=0.2, l=2).to(latent.device)  # [N, L]

        l_penalty = latent.norm(dim=-1).mean()

        q = latent[mask]  # [Lmask, C]
        q, l_diversity = self.quantizer(q)  # [Lmask, Cout]

        c = self.pre_extract(latent)
        c = c * ~mask[..., None] + self.masked_code * mask[..., None]

        c = c.transpose(1, 2)  # [N, L, C] -> [N, C, L]
        c = self.pos_emb(c)
        c = c.transpose(1, 2)  # [N, C, L] -> [N, L, C]

        c = self.context_model(c)  # [N, L, Cout]
        c = c[mask]  # [Lmask, Cout]

        sim = nn.functional.cosine_similarity(c[:, None], q[None, :], dim=-1)  # [Lmask(c), Lmask(q)]
        temperature = 0.1
        sim = sim / temperature
        l_contrastive = nn.functional.cross_entropy(sim, torch.arange(sim.shape[-1], device=sim.device))

        loss = l_contrastive + 0.1 * l_diversity + 10 * l_penalty

        # for logging
        self.l_contrastive = l_contrastive.detach()
        self.l_diversity = l_diversity.detach()
        self.l_penalty = l_penalty.detach()

        return loss


def predictor_small():
    return ContextPredictor(
        channels=(32, 48, 72, 108, 160, 224, 320),
        res_num=(1,) * 7, down_sample_scale=3,
        groups=(4, 4, 6, 6, 8, 8, 8),
        context_channel=512, context_num_layers=8, context_num_heads=8,
        codebook_group=2, codebook_entry=192,
        out_ch=160,
    )
