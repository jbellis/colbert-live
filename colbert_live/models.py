from abc import ABC, abstractmethod
from typing import List

import torch
from PIL.Image import Image
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score_packed
from colpali_engine.models import ColPali, ColPaliProcessor

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="colbert.utils.amp")


class Model(ABC):
    @abstractmethod
    def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into a tensor of embeddings.

        Args:
            q: The query string to encode.

        Returns:
            A 2D tensor of query embeddings.
            The tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).
        """
        pass

    @abstractmethod
    def encode_doc(self, chunks: List[str]) -> List[torch.Tensor]:
        """
        Encode a batch of document chunks into tensors of embeddings.

        Args:
            chunks: A list of content strings to encode.

        Returns:
            A list of 2D tensors of embeddings, one for each input chunk.
            Each tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).
        """
        pass

    @abstractmethod
    def score(self, Q: torch.Tensor, D_packed: torch.Tensor, D_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate ColBERT scores for given query and document embeddings.

        Args:
            Q: 2D Query embeddings tensor.
            D_packed: Packed document embeddings tensor.
            D_lengths: Tensor of document lengths.

        Returns:
            A tensor of ColBERT scores.
        """
        pass

    @property
    @abstractmethod
    def use_gpu(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @staticmethod
    def from_name_or_path(name_or_path: str, **kwargs):
        if 'colbert' in name_or_path.lower():
            return ColbertModel(name_or_path, **kwargs)
        elif 'colpali' in name_or_path.lower():
            return ColpaliModel(name_or_path, **kwargs)
        else:
            raise ValueError(
                f"Unknown model: {name_or_path}. You can manually instantiate an instance of ColbertModel or ColpaliModel.")


class ColbertModel(Model):
    def __init__(self, model_name: str, tokens_per_query: int = 32):
        self.config = ColBERTConfig(checkpoint=model_name, query_maxlen=tokens_per_query)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config)
        self.encoder = CollectionEncoder(self.config, self.checkpoint)

    def encode_query(self, q: str) -> torch.Tensor:
        return self.checkpoint.queryFromText([q])[0]

    def encode_doc(self, chunks: List[str]) -> List[torch.Tensor]:
        input_ids, attention_mask = self.checkpoint.doc_tokenizer.tensorize(chunks)
        D, mask = self.checkpoint.doc(input_ids, attention_mask, keep_dims='return_mask')

        embeddings_list = []
        for i in range(len(chunks)):
            Di = D[i]
            maski = mask[i].squeeze(-1).bool()
            Di = Di[maski]  # Keep only non-padded embeddings
            embeddings_list.append(Di)

        return embeddings_list

    def score(self, Q: torch.Tensor, D_packed: torch.Tensor, D_lengths: torch.Tensor) -> torch.Tensor:
        # colbert_score_packed expects a 3D query tensor even though it only operates on a single query
        return colbert_score_packed(Q.unsqueeze(0), D_packed, D_lengths, config=self.config)

    @property
    def use_gpu(self):
        return self.checkpoint.use_gpu

    @property
    def dim(self):
        return self.config.dim


class ColpaliModel(Model):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.colpali = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if device == "cuda" else None,
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def encode_query(self, q: str) -> torch.Tensor:
        with torch.no_grad():
            batch = self.processor.process_queries([q])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.colpali(**batch)

        return embeddings[0]

    def encode_doc(self, images: List[Image]) -> List[torch.Tensor]:
        with torch.no_grad():
            batch = self.processor.process_images(images)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            embeddings = self.colpali(**batch)

        return list(torch.unbind(embeddings))

    def score(self, Q: torch.Tensor, D_packed: torch.Tensor, D_lengths: torch.Tensor) -> torch.Tensor:
        # Unpack D_packed into a list of tensors based on D_lengths
        D = torch.split(D_packed, D_lengths.tolist())

        # Compute scores using the processor's score method
        # ColPaLi scoring is a batch interface, so unsqueeze here
        scores = self.processor.score(Q.unsqueeze(0), D)

        return scores.squeeze(0)  # Remove batch dimension

    @property
    def use_gpu(self):
        return self.device == "cuda"

    @property
    def dim(self):
        return self.colpali.dim
