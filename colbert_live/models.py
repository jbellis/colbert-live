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

    def score(self, Q: torch.Tensor, D: list[torch.Tensor]) -> torch.Tensor:
        """
        Calculate ColBERT scores for given query and document embeddings.

        Args:
            Q: 2D Query embeddings tensor.
            D: list of 2D document embeddings tensors.

        Returns:
            A tensor of ColBERT scores.
        """
        Q = self.to_device(Q.unsqueeze(0))  # Add batch dimension and move to device
        D = [self.to_device(d.to(Q.dtype)) for d in D]  # Move passage embeddings to device

        # Pad the passage embeddings to the same length
        D_padded = torch.nn.utils.rnn.pad_sequence(D, batch_first=True, padding_value=0)

        # Compute scores using einsum
        scores = torch.einsum("bnd,csd->bcns", Q, D_padded).max(dim=3)[0].sum(dim=2)

        return scores.squeeze(0).to(torch.float32)  # Remove batch dimension and convert to float32

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

    def to_device(self, T: torch.Tensor):
        return T.to(_get_module_device(self.checkpoint))

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        return f"ColbertModel(model={self.colpali.model_name}, dim={self.dim}, device={_get_module_device(self.colpali)})"


class ColpaliModel(Model):
    def __init__(self, model_name: str = 'vidore/colqwen2-v0.1'):
        # for `score()`
        if ColBERTConfig().total_visible_gpus == 0:
            ColBERT.try_load_torch_extensions(False)

        # processor is optional (useful for the vidore benchmark: can still compute scores w/o a model)
        if not model_name:
            self.processor = None
            return

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

    def to_device(self, T: torch.Tensor):
        return T.to(_get_module_device(self.colpali))

    @property
    def dim(self):
        return self.colpali.dim

    def __str__(self):
        return f"ColpaliModel(model={self.colpali.model_name}, dim={self.dim}, device={_get_module_device(self.colpali)})"
