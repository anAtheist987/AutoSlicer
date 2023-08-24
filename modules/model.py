from collections import OrderedDict

import torch
from torch import nn

from typing import Sequence, Union, Optional


def normalization(channel):
    return nn.BatchNorm1d(channel)


def activation():
    return nn.GELU()


def zero_module(module: torch.nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Shortcut(nn.Sequential):
    def forward(self, x):
        return x + super(Shortcut, self).forward(x)


class ResNeXt1d(nn.Module):
    def __init__(self, in_ch, kernel_size=3, padding: Union[str, int] = 'same', groups=4):
        super(ResNeXt1d, self).__init__()

        self.layers = Shortcut(
            normalization(in_ch),
            activation(),
            nn.Conv1d(in_ch, in_ch // 2, kernel_size=1, padding=0),

            normalization(in_ch // 2),
            activation(),
            nn.Conv1d(in_ch // 2, in_ch // 2, kernel_size=kernel_size, padding=padding, groups=groups),

            normalization(in_ch // 2),
            activation(),
            zero_module(nn.Conv1d(in_ch // 2, in_ch, kernel_size=1, padding=0)),

            nn.Dropout(0.2)
        )

    def forward(self, x):
        """

        :param x: [N, C, L]
        :return: [N, C, L]
        """
        return self.layers(x)


class UnaryGRU(nn.GRU):
    def forward(self, x, hx=None):
        x, h_n = super().forward(x, hx)
        return x


class BiGRUs(nn.Module):
    def __init__(self, channels, dropout=0., num_layers=1, shortcut=True):
        super(BiGRUs, self).__init__()
        if shortcut:
            self.grus = nn.Sequential()
            for i in range(num_layers):
                self.grus.add_module(f"gru{i}", Shortcut(
                    UnaryGRU(channels, channels // 2, bidirectional=True, batch_first=True, num_layers=1),
                    nn.Dropout(dropout),
                ))
        else:
            self.grus = UnaryGRU(channels, channels // 2, bidirectional=True, dropout=dropout, batch_first=True,
                                 num_layers=num_layers)
        self.shortcut = shortcut

    def forward(self, x):
        return self.grus(x)


class SincConvFast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding: Union[str, int] = 0, dilation=1, bias=False, groups=1,
                 min_low_hz=50, min_band_hz=50, filter_type='band_pass'):

        super(SincConvFast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = torch.tensor(30)
        high_hz = torch.tensor(self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz))

        mel = torch.linspace(self.to_mel(low_hz),
                             self.to_mel(high_hz),
                             self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz = nn.Parameter(hz[:-1].view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz = nn.Parameter(torch.diff(hz).view(-1, 1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n_lin / self.kernel_size)  # hamming

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = 2 * torch.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

        self.filter_type = filter_type

    def forward(self, waveforms: torch.Tensor):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        if self.filter_type == 'band_pass':
            return self.band_pass(waveforms)
        elif self.filter_type == 'band_stop':
            return self.band_stop(waveforms)
        elif self.filter_type == 'both':
            return self.band_pass(self.band_stop(waveforms))

    def get_band_pass_filter(self, low, high):
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)
        return filters

    def get_band_stop_filter(self, low, high):
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_stop_left = ((torch.sin(f_times_t_low) - torch.sin(f_times_t_high) + torch.sin(self.n_ / 2)) / (
                self.n_ / 2)) * self.window  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_stop_center = 2 * band.view(-1, 1)
        band_stop_right = torch.flip(band_stop_left, dims=[1])

        band_stop = torch.cat([band_stop_left, band_stop_center, band_stop_right], dim=1)

        band_stop = band_stop / (2 * band[:, None])

        filters = (band_stop).view(
            self.out_channels, 1, 1, self.kernel_size)
        return filters

    def band_pass(self, waveforms: torch.Tensor):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window = self.window.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return nn.functional.conv1d(waveforms, self.filters, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation,
                                    bias=None, groups=1)

    def band_stop(self, waveforms, layer_norm=None):
        self.n_ = self.n_.to(waveforms.device)

        self.window = self.window.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_stop_left = ((torch.sin(f_times_t_low) - torch.sin(f_times_t_high) + torch.sin(self.n_ / 2)) / (
                self.n_ / 2)) * self.window  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_stop_center = 2 * band.view(-1, 1)
        band_stop_right = torch.flip(band_stop_left, dims=[1])

        band_stop = torch.cat([band_stop_left, band_stop_center, band_stop_right], dim=1)

        band_stop = band_stop / (2 * band[:, None])

        self.filters = (band_stop).view(
            self.out_channels, 1, 1, self.kernel_size)

        for idx in range(self.out_channels):
            waveforms = nn.functional.conv1d(waveforms, self.filters[idx], stride=self.stride,
                                             padding=self.kernel_size // 2, dilation=self.dilation, bias=None, groups=1)
            waveforms = layer_norm(waveforms) if layer_norm is not None else waveforms
        return waveforms


class Encoder(nn.Module):
    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),
            groups: Union[int, Sequence[int]] = 4,
            res_num: Union[int, Sequence[int]] = 1,
            down_sample_scale: Union[int, Sequence[int]] = 3,
    ):
        super(Encoder, self).__init__()
        if isinstance(groups, int):
            groups = (groups,) * len(channels)
        if isinstance(res_num, int):
            res_num = (res_num,) * len(channels)
        if isinstance(down_sample_scale, int):
            down_sample_scale = (down_sample_scale,) * (len(channels) - 1)

        self.stem = SincConvFast(out_channels=channels[0], kernel_size=65, padding='same', sample_rate=8000)

        self.down_samples = nn.Sequential()
        for level in range(len(channels)):
            layer = nn.Sequential(OrderedDict([
                (f"res{i}", ResNeXt1d(channels[level], kernel_size=5, groups=groups[level]))
                for i in range(res_num[level])
            ]))
            if level < len(channels) - 1:
                layer.add_module("downsample", nn.Conv1d(
                    channels[level], channels[level + 1],
                    kernel_size=down_sample_scale[level], stride=down_sample_scale[level],
                ))
            self.down_samples.add_module(f"layer{level}", layer)

        self.final = nn.Sequential(
            normalization(channels[-1]),
        )

    def forward(self, x):
        """
        :param x: [N, L]
        :return: [N, C, L // scale ** (layer_num - 1)]
        """

        x = x[:, None, :]  # [N, 1, L]
        x = self.stem(x)
        x = self.down_samples(x)
        x = self.final(x)
        return x


class Slicer(nn.Module):
    """
    CRNN
    multi-scale conv1d on original sound wave
    BiGRU
    """

    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),
            groups: Union[int, Sequence[int]] = 4,
            res_num: Union[int, Sequence[int]] = 1,
            down_sample_scale: Union[int, Sequence[int]] = 3,
            context_channel=192, context_num_layers=4,
            out_ch=1,
    ):
        super(Slicer, self).__init__()

        self.encoder = Encoder(channels=channels, res_num=res_num, groups=groups, down_sample_scale=down_sample_scale)

        self.context_model = nn.Sequential(OrderedDict([
            ("proj_in", nn.Linear(channels[-1], context_channel)),
            ("norm", nn.LayerNorm(context_channel)),
            ("act", activation()),
            ("BiGRUs", BiGRUs(context_channel, dropout=0.2, num_layers=context_num_layers)),
        ]))

        self.head = nn.Sequential(
            nn.Linear(context_channel, out_ch),
        )

    def forward(self, x, target):
        """
        :param x: [N, L]
        :param target: [N, L // scale ** (layer_num - 1), 1]
        :return: [N, Cout, L // scale ** (layer_num - 1)]
        """

        x = self.encoder(x)  # [N, L] -> [N, C, Lout]

        x = x.transpose(1, 2)  # [N, C, Lout] -> [N, Lout, C]
        x = self.context_model(x)

        x = self.head(x)

        return nn.functional.binary_cross_entropy_with_logits(x, target)


def slicer_small():
    return Slicer(
        channels=(32, 48, 72, 108, 160, 224, 320),
        res_num=(1,) * 7, down_sample_scale=3,
        groups=(4, 4, 6, 6, 8, 8, 8),
        context_channel=512, context_num_layers=4,
        out_ch=1,
    )
