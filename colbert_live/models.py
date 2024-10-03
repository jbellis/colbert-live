from abc import ABC, abstractmethod

import torch
from PIL.Image import Image
from colbert import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT, colbert_score_packed
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor

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
    def encode_doc(self, chunks: list[str|Image]) -> list[torch.Tensor]:
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

    @abstractmethod
    def to_device(self, T: torch.Tensor):
        """
        Copy a tensor to the device used by this model.  (Used when loading from the database.)
        """
        pass

    @property
    @abstractmethod
    def dim(self):
        pass


def _get_module_device(module):
    return next(module.parameters()).device


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

    def score(self, Q: torch.Tensor, D_packed: torch.Tensor, D_lengths: torch.Tensor) -> torch.Tensor:
        # colbert_score_packed expects a 3D query tensor even though it only operates on a single query
        return colbert_score_packed(Q.unsqueeze(0), D_packed, D_lengths, config=self.config)

    def to_device(self, T: torch.Tensor):
        return T.to(_get_module_device(self.checkpoint))

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        return f"ColbertModel(model={self.colpali.model_name}, dim={self.dim}, device={_get_module_device(self.colpali)})"


class ColpaliModel(Model):
    def __init__(self, model_name: str = 'vidore/colqwen2-v0.1'):
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
        if ColBERTConfig().total_visible_gpus == 0:
            ColBERT.try_load_torch_extensions(False)


    def encode_query(self, q: str) -> torch.Tensor:
        with torch.no_grad():
            batch = self.processor.process_queries([q])
            batch = {k: self.to_device(v) for k, v in batch.items()}
            embeddings = self.colpali(**batch)

        return embeddings[0]

    def encode_doc(self, images: list[Image]) -> list[torch.Tensor]:
        with torch.no_grad():
            batch = self.processor.process_images(images)
            batch = {k: self.to_device(v) for k, v in batch.items()}
            raw_embeddings = self.colpali(**batch)

        # Discard zero vectors from the embeddings tensor
        return [emb[emb.norm(dim=-1) > 0] for emb in raw_embeddings]

    def score(self, Q: torch.Tensor, D_packed: torch.Tensor, D_lengths: torch.Tensor) -> torch.Tensor:
        # Use colbert's scoring method because colbert-live operates in the float32 domain instead of bfloat16
        # We don't pass a config object because the default is good enough for what we need
        # (which is just reading total_visible_gpus to decide whether to call the C++ extension)
        return colbert_score_packed(Q.unsqueeze(0), D_packed, D_lengths)

    def to_device(self, T: torch.Tensor):
        return T.to(_get_module_device(self.colpali))

    @property
    def dim(self):
        return self.colpali.dim

    def __str__(self):
        return f"ColpaliModel(model={self.colpali.model_name}, dim={self.dim}, device={_get_module_device(self.colpali)})"
