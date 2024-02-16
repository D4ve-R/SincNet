
""" SincNet model """
from functools import lru_cache
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SincNetFilterConvLayer(nn.Module):
    """SincNet fast convolution filter layer"""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate=16000, 
                stride=1, padding=0, dilation=1, min_low_hz=50, min_band_hz=50, 
                in_channels=1, requires_grad=False):
        """
        Args:
            out_channels : `int` number of filters.
            kernel_size : `int` filter length.
            sample_rate : `int`, optional sample rate. Defaults to 16000.
        """
        super(SincNetFilterConvLayer, self).__init__()

        if in_channels != 1:
            raise ValueError(f"SincNetFilterConvLayer only support in_channels = 1, was in_channels = {in_channels}")

        self._out_channels = out_channels
        self._kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self._kernel_size += 1 # Forcing the filters to be odd
            
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._sample_rate = sample_rate
        self._min_low_hz = min_low_hz
        self._min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self._sample_rate / 2 - (self._min_low_hz + self._min_band_hz)
        mel = np.linspace(
            2595 * np.log10(1 + low_hz / 700), # Convert Hz to Mel
            2595 * np.log10(1 + high_hz / 700), # Convert Hz to Mel
            self._out_channels // 2 + 1
        )
        hz = 700 * (10 ** (mel / 2595) - 1) # Convert Mel to Hz
        
        self._low_hz = nn.Parameter(
            torch.Tensor(hz[:-1]).view(-1, 1),
            requires_grad=requires_grad
        )
        self._band_hz = nn.Parameter(
            torch.Tensor(np.diff(hz)).view(-1, 1),
            requires_grad=requires_grad
        )
        self.register_buffer(
            "_window", 
            torch.from_numpy(np.hamming(self._kernel_size)[: self._kernel_size // 2]).float()
        )
        self.register_buffer(
            "_n",
            (2* np.pi * torch.arange(-(self._kernel_size // 2), 0.0).view(1, -1) / self._sample_rate)
        )

    @property
    @lru_cache(maxsize=1)
    def filters(self) -> torch.Tensor:
        low = self._min_low_hz + torch.abs(self._low_hz)
        high = torch.clamp(low + self._min_band_hz + torch.abs(self._band_hz), self._min_low_hz, self._sample_rate/2)
        band = (high-low)[:,0]

        f_times_t_low = torch.matmul(low, self._n)
        f_times_t_high = torch.matmul(high, self._n)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self._n/2))*self._window
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        band_pass = band_pass / (2*band[:,None])
        return band_pass.view(self._out_channels, 1, self._kernel_size)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms : (batch_size, 1, n_samples) batch of waveforms.
        
        Returns:
            features : (batch_size, out_channels, n_samples_out) batch of sinc filters activations.
        """
        return F.conv1d(waveforms, self.filters, stride=self._stride,
                        padding=self._padding, dilation=self._dilation,
        ).abs_() # https://github.com/mravanelli/SincNet/issues/4

class SincNet(nn.Module):
    """SincNet"""

    def __init__(
        self,  
        num_sinc_filters: int = 80,
        sinc_filter_length: int = 251,
        num_conv_filters: int = 60,
        conv_filter_length: int = 5,
        pool_kernel_size: int = 3,
        pool_stride: int = 3,
        sample_rate: int = 16000,
        sinc_filter_stride: int = 10,
        sinc_filter_padding: int = 0,
        sinc_filter_dilation: int = 1,
        min_low_hz: int = 50,
        min_band_hz: int = 50,
        sinc_filter_in_channels: int = 1,
        num_wavform_channels: int = 1,
    ):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError(f"SincNet only supports 16kHz audio (sample_rate = 16000), was sample_rate = {sample_rate}")

        self.wav_norm1d = nn.InstanceNorm1d(num_wavform_channels, affine=True)

        self.conv1d = nn.ModuleList([
            SincNetFilterConvLayer(
                num_sinc_filters, 
                sinc_filter_length, 
                sample_rate=sample_rate, 
                stride=sinc_filter_stride,
                padding=sinc_filter_padding,
                dilation=sinc_filter_dilation,
                min_low_hz=min_low_hz,
                min_band_hz=min_band_hz,
                in_channels=sinc_filter_in_channels,
            ),
            nn.Conv1d(num_sinc_filters, num_conv_filters, conv_filter_length),
            nn.Conv1d(num_conv_filters, num_conv_filters, conv_filter_length),
        ])
        self.pool1d = nn.ModuleList([
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
            nn.MaxPool1d(pool_kernel_size, stride=pool_stride),
        ])
        self.norm1d = nn.ModuleList([
            nn.InstanceNorm1d(num_sinc_filters, affine=True),
            nn.InstanceNorm1d(num_conv_filters, affine=True),
            nn.InstanceNorm1d(num_conv_filters, affine=True),
        ])

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms : (batch, channel, sample)
        """
        outputs = self.wav_norm1d(waveforms)

        for _, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            outputs = conv1d(outputs)
            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs
