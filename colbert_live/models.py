from abc import ABC
from typing import List

import torch
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig


class Model(ABC):
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

    @property
    def use_gpu(self):
        pass


class ColbertModel(Model):
    def __init__(self, model_name: str, tokens_per_query: int):
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

    @property
    def use_gpu(self):
        return self.checkpoint.use_gpu
