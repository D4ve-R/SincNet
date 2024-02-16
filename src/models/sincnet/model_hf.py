from transformers import PreTrainedModel, AutoConfig, AutoModel
from .model import SincNet
from .config import SincNetConfig

class SincNetModel(PreTrainedModel):
    config_class = SincNetConfig
    base_model_prefix = "sincnet"

    def __init__(self, config: SincNetConfig):
        super().__init__(config)

        self.model = SincNet(
            sinc_filter_stride=config.stride,
            num_sinc_filters=config.num_sinc_filters,
            sinc_filter_length=config.sinc_filter_length,
            num_conv_filters=config.num_conv_filters,
            conv_filter_length=config.conv_filter_length,
            pool_kernel_size=config.pool_kernel_size,
            pool_stride=config.pool_stride,
            sample_rate=config.sample_rate,
        )
    
    def forward(self, waveforms):
        return self.model(waveforms)

AutoConfig.register('sincnet', SincNetConfig)
AutoModel.register(SincNetConfig, SincNetModel)
