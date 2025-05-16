from diffusers import PixArtSigmaPipeline
from models.pipeline_pixart_sigma import MXPixArtSigmaPipeline

print("base:", PixArtSigmaPipeline._component_names)
print("custom:", MXPixArtSigmaPipeline._component_names)





