"""KugelAudio model components."""

from .kugelaudio_model import (
    KugelAudioModel,
    KugelAudioPreTrainedModel,
    KugelAudioForConditionalGeneration,
)
from .kugelaudio_inference import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioCausalLMOutputWithPast,
    KugelAudioGenerationOutput,
)
from .tokenizer import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioSemanticTokenizerModel,
    KugelAudioTokenizerEncoderOutput,
)
from .diffusion_head import KugelAudioDiffusionHead
from .conv_layers import (
    RMSNorm,
    ConvRMSNorm,
    ConvLayerNorm,
    SConv1d,
    SConvTranspose1d,
)
from .low_vram import LowVRAMInferenceWrapper, load_model_low_vram, load_model_quantized

__all__ = [
    # Main models
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioForConditionalGenerationInference",
    # Low-VRAM inference
    "LowVRAMInferenceWrapper",
    "load_model_low_vram",
    "load_model_quantized",
    # Outputs
    "KugelAudioCausalLMOutputWithPast",
    "KugelAudioGenerationOutput",
    # Tokenizers
    "KugelAudioAcousticTokenizerModel",
    "KugelAudioSemanticTokenizerModel",
    "KugelAudioTokenizerEncoderOutput",
    # Components
    "KugelAudioDiffusionHead",
    "RMSNorm",
    "ConvRMSNorm",
    "ConvLayerNorm",
    "SConv1d",
    "SConvTranspose1d",
]
