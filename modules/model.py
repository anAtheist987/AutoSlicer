import torch
from torch import nn

from typing import Sequence, Union


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


class ResBlk(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, padding: Union[str, int] = 'same'):
        super(ResBlk, self).__init__()

        if out_ch is None:
            out_ch = in_ch

        self.layers = nn.Sequential(
            normalization(in_ch),
            activation(),
            nn.Conv1d(in_ch, out_ch // 4, kernel_size=1, padding=0),
            normalization(out_ch // 4),
            activation(),
            nn.Conv1d(out_ch // 4, out_ch // 4, kernel_size=kernel_size, padding=padding),
            normalization(out_ch // 4),
            activation(),
            nn.Conv1d(out_ch // 4, out_ch, kernel_size=1, padding=0),

        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """

        :param x: [B, Cin, L]
        :return: [B, Cout, L]
        """
        shortcut = x
        x = self.layers(x)
        x = self.dropout(x)

        return x + shortcut


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0., num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x


class TransBlk(nn.Module):
    pass


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

    def forward(self, waveforms):
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

    def band_pass(self, waveforms):
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
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),  # down sample 6 times (256ms in 16k SR)
            res_num=(1,) * 7,
    ):
        super(Encoder, self).__init__()

        self.stem = SincConvFast(out_channels=channels[0], kernel_size=65, padding='same', sample_rate=8000)

        self.down_samples = nn.ModuleList()
        for i in range(len(channels)):
            layer = nn.Sequential(
                *([ResBlk(channels[i], kernel_size=5)] * res_num[i])
            )
            if i < len(channels) - 1:
                layer.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=3))
            self.down_samples.append(layer)

    def forward(self, x):
        """
        :param x: [B, L]
        :return: [B, C, L // 2 ** 12]
        """

        x = x[:, None, :]  # [B, 1, L]

        x = self.stem(x)

        for down in self.down_samples:
            x = down(x)

        return x


class Slicer(nn.Module):
    """
    CRNN
    multi-scale conv1d on original sound wave
    BiGRU
    """

    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),  # down sample 6 times (256ms in 16k SR)
            res_num=(1,) * 7,
            out_ch=1,
            rnn_channel=192,
    ):
        super(Slicer, self).__init__()

        self.encoder = Encoder(channels=channels, res_num=res_num)

        self.gru = BiGRU(channels[-1], rnn_channel // 2, dropout=0.2, num_layers=4)

        self.head = nn.Sequential(
            normalization(rnn_channel),
            activation(),
            nn.Conv1d(rnn_channel, out_ch, kernel_size=1),
        )

    def forward(self, x, label):
        """
        :param x: [B, L]
        :param label: [B, L]
        :return: [B, Cout, L]
        """

        x = self.encoder(x)  # [B, L] -> [B, C, L]

        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        x = self.gru(x)

        x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        x = self.head(x)

        return torch.nn.functional.binary_cross_entropy_with_logits(x, label)


class Pretrain(nn.Module):

    def __init__(
            self,
            channels: Sequence[int] = (32, 48, 72, 96, 128, 160, 192),  # down sample 6 times (256ms in 16k SR)
            res_num=(1,) * 7,
            out_ch=1,
    ):
        super(Pretrain, self).__init__()

        self.encoder = Encoder(channels=channels, res_num=res_num)

        self.head = nn.Sequential(

        )

