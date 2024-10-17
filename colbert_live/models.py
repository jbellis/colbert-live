import warnings
from abc import ABC, abstractmethod

import torch
from PIL.Image import Image
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor

warnings.filterwarnings("ignore", category=FutureWarning, module="colbert.utils.amp")


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xor = a ^ b
    return torch.popcnt(xor).sum(dim=-1)


class Model(ABC):
    @abstractmethod
    def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into a tensor of embeddings.

        Args:
            q: The query string to encode.

        Returns:
            A 2D float32 tensor of query embeddings.
            The tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).
        """
        pass

    @abstractmethod
    def encode_doc(self, chunks: list[str|Image]) -> list[torch.Tensor]:
        """
        Encode a batch of document chunks into tensors of embeddings.

        Args:
            chunks: A list of content strings to encode.

        Returns:
            A list of 2D float32 tensors of embeddings, one for each input chunk.
            Each tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).
        """
        pass

    def score(self, Q: torch.Tensor, D: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculate ColBERT scores for given query and document embeddings.

        Args:
            Q: 2D Query embeddings tensor.
            D: list of 2D document embeddings tensors.
        Note: Q and D may be int64 tensors representing binary-quantized embeddings, or float32 tensors.
        Both must be one or the other.

        Returns:
            A 1D float32 tensor of ColBERT scores.
        """
        Q = Q.unsqueeze(0).cpu()  # Add batch dimension and move to device
        D = [d.to(Q.dtype).cpu() for d in D]  # Move passage embeddings to device

        if Q.dtype == torch.float32:
            # Original float32 maxsim scoring method
            D = [d.to(Q.dtype) for d in D]
            D_padded = torch.nn.utils.rnn.pad_sequence(D, batch_first=True, padding_value=0)
            scores = torch.einsum("bnd,csd->bcns", Q, D_padded).max(dim=3)[0].sum(dim=2)
        elif Q.dtype == torch.int64:
            # Binary-quantized scoring method using Hamming distance
            max_doc_length = max(d.shape[0] for d in D)
            D_padded = torch.zeros((len(D), max_doc_length, Q.shape[-1]), dtype=torch.int64)
            for i, d in enumerate(D):
                D_padded[i, :d.shape[0]] = d

            # Compute Hamming distances
            distances = hamming_distance(Q.unsqueeze(1).unsqueeze(1), D_padded.unsqueeze(0))

            # Convert distances to similarity scores (lower distance = higher similarity)
            max_distance = Q.shape[-1] * 64  # Maximum possible Hamming distance (64 bits per int64)
            similarities = max_distance - distances

            # Aggregate similarities
            scores = similarities.max(dim=3)[0].sum(dim=2)
        else:
            raise ValueError("Input tensors must be either float32 or int64")

        return scores.squeeze(0).to(torch.float32)  # Remove batch dimension and convert to float32

    @property
    @abstractmethod
    def dim(self):
        pass


class ColbertModel(Model):
    def __init__(self, model_name: str = 'answerdotai/answerai-colbert-small-v1', tokens_per_query: int = 32):
        self.config = ColBERTConfig(checkpoint=model_name, query_maxlen=tokens_per_query)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config)
        self.encoder = CollectionEncoder(self.config, self.checkpoint)
        # TODO is there a better way to get the output dimension?
        self._dim = len(self.encode_query("foo")[0])

    def encode_query(self, q: str) -> torch.Tensor:
        return self.checkpoint.queryFromText([q])[0]

    def encode_doc(self, chunks: list[str]) -> list[torch.Tensor]:
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
    def dim(self):
        return self._dim

    def __str__(self):
        return f"ColbertModel(config={self.config}, dim={self.dim})"


class ColpaliModel(Model):
    def __init__(self, model_name: str = 'vidore/colqwen2-v0.1'):
        # load processor
        if 'qwen' in model_name:
            cls = ColQwen2
            prs = ColQwen2Processor
        else:
            cls = ColPali
            prs = ColPaliProcessor
        self.colpali = cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = prs.from_pretrained(model_name)


    def encode_query(self, q: str) -> torch.Tensor:
        with torch.no_grad():
            batch = self.processor.process_queries([q])
            batch = {k: v.cpu() for k, v in batch.items()}
            embeddings = self.colpali(**batch)

        return embeddings[0]

    def encode_doc(self, images: list[Image]) -> list[torch.Tensor]:
        with torch.no_grad():
            batch = self.processor.process_images(images)
            batch = {k: v.cpu() for k, v in batch.items()}
            raw_embeddings = self.colpali(**batch)

        # Discard zero vectors from the embeddings tensor
        return [emb[emb.norm(dim=-1) > 0] for emb in raw_embeddings]

    @property
    def dim(self):
        return self.colpali.dim

    def __str__(self):
        return f"ColpaliModel(model={self.colpali.model_name}, dim={self.dim})"
