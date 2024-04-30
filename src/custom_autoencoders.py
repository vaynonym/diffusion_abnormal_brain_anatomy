import torch
from src.synthseg_masks import synthseg_classes
from src.util import device
from src.logging_util import LOGGER
from generative.networks.nets import AutoencoderKL
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class IAutoencoder(nn.Module):

    def forward(self, x: Tensor):
        pass
    
    def reconstruct(self, x: Tensor) -> Tensor:
        pass

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def decode(self, z: Tensor) -> Tensor:
        pass

    def sampling(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        pass
    
    def encode_stage_2_inputs(self, x: Tensor) -> Tensor:
        pass
    
    def decode_stage_2_outputs(self, z: Tensor) -> Tensor:
        pass

class EmbeddingWrapper(IAutoencoder):
    def __init__(self,
                 base_autoencoder: AutoencoderKL,
                 vocabulary_size,
                 embedding_dim,
                 ) -> None:
        super().__init__()
        self.autoencoder = base_autoencoder
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim).to(device)
        LOGGER.info(f"Embedding accepts indices up to {vocabulary_size}")

    
    def prepare_input(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        assert len(list(original_shape)) == 5
        x = x[:, 0]
        x = self.embedding(x)
        x = torch.movedim(x, len(list(x.shape)) - 1, 1)
        assert original_shape[0] == x.shape[0]
        assert x.shape[1] == self.embedding.embedding_dim
        assert original_shape[2] == x.shape[2]
        assert original_shape[3] == x.shape[3]
        assert original_shape[4] == x.shape[4]
        return x

    
    def forward(self, x: Tensor) -> Tensor:
        x_emb = self.prepare_input(x)
        return self.autoencoder(x_emb)
    
    def reconstruct(self, x: Tensor) -> Tensor:
        x_emb = self.prepare_input(x)
        return self.autoencoder.reconstruct(x_emb)
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_emb = self.prepare_input(x)
        return self.autoencoder.encode(x_emb)
    
    def decode(self, z: Tensor) -> Tensor:
        return self.autoencoder.decode(z)
    
    def sampling(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        return self.autoencoder.sampling(z_mu, z_sigma)
    
    def encode_stage_2_inputs(self, x: Tensor) -> Tensor:
        x_emb = self.prepare_input(x)
        return self.autoencoder.encode_stage_2_inputs(x_emb)
    
    def decode_stage_2_outputs(self, z: Tensor) -> Tensor:
        return self.autoencoder.decode_stage_2_outputs(z)


def create_embedding_autoencoder(*args, **kwargs):
    base_autoencoder = AutoencoderKL(*args, **kwargs)
    autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)
    return autoencoder