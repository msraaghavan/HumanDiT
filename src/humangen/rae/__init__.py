"""RAEv2 representation autoencoder (frozen DINOv3-L encoder + ViT-XL pixel decoder)."""

from humangen.rae.autoencoder import LATENT_SHAPE, RAE, ensure_weights

__all__ = ["LATENT_SHAPE", "RAE", "ensure_weights"]
