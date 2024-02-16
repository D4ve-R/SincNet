from transformers import PretrainedConfig

class SincNetConfig(PretrainedConfig):
    model_type = "sincnet"
    
    def __init__(
        self,
        stride: int = 10,
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
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.stride = stride
        self.num_sinc_filters = num_sinc_filters
        self.sinc_filter_length = sinc_filter_length
        self.num_conv_filters = num_conv_filters
        self.conv_filter_length = conv_filter_length
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.sinc_filter_stride = sinc_filter_stride
        self.sinc_filter_padding = sinc_filter_padding
        self.sinc_filter_dilation = sinc_filter_dilation
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.sinc_filter_in_channels = sinc_filter_in_channels
        self.num_wavform_channels = num_wavform_channels
        super().__init__(**kwargs)   
