from collections import OrderedDict

import torch
from torch import nn

from typing import Sequence, Union, Optional

from .model import Slicer, activation, normalization


class ContextPredictor(Slicer):
    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),
            groups: Union[int, Sequence[int]] = 4,
            res_num: Union[int, Sequence[int]] = 1,
            down_sample_scale: Union[int, Sequence[int]] = 3,
            context_channel=192, context_num_layers=4, context_num_heads=4,
            out_ch=1,
    ):
        super(ContextPredictor, self).__init__(
            channels=channels, groups=groups, res_num=res_num, down_sample_scale=down_sample_scale,
            context_channel=context_channel, context_num_layers=0, out_ch=out_ch,
        )

        self.context_model = nn.Sequential(OrderedDict([
            ("proj_in", nn.Linear(channels[-1], context_channel)),
            ("norm", nn.LayerNorm(context_channel)),
            ("act", activation()),
        ]))

    def forward(self, x, target):

        x = self.encoder(x)  # [N, L] -> [N, C, Lout]

        x = x.permute(0, 2, 1)  # [N, C, Lout] -> [N, Lout, C]
        x = self.context_model(x)

        x = self.head(x)

        return torch.nn.functional.binary_cross_entropy(x, target)


def predictor_small():
    return ContextPredictor(
        channels=(32, 48, 72, 108, 160, 224, 320),
        res_num=(1,) * 7, down_sample_scale=3,
        groups=(4, 4, 6, 6, 8, 8, 8),
        context_channel=512, context_num_layers=8, context_num_heads=8,
        out_ch=256,
    )
