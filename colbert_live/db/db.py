from abc import ABC, abstractmethod
from typing import List, Any, Iterable, Tuple

import torch


class DB(ABC):
    @abstractmethod
    def query_ann(self, embeddings: torch.Tensor, limit: int) -> list[list[tuple[Any, float]]]:
        """
        Perform an approximate nearest neighbor (ANN) search of the ColBERT embeddings.
        This informs the decision of which documents to fetch for full ColBERT scoring.

        Args:
            embeddings: A 2D tensor of ColBERT embeddings to compare against.
            limit: The maximum number of results to return for each embedding.

        Returns:
            A list of lists, one per embedding, where each inner list contains tuples of (PK, similarity)
            for the chunks closest to each query embedding.
        """
        pass

    @abstractmethod
    def query_chunks(self, pks: list[Any]) -> Iterable[torch.Tensor]:
        """
        Retrieve all ColBERT embeddings for specific chunks so that ColBERT scores
        can be computed.

        Args:
            pks: A list of primary keys (of any object type) identifying the chunks.

        Returns:
            An interable of 2D tensors representing the ColBERT embeddings for each of the specified chunks.
        """
        pass
