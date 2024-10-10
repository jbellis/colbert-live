from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch


# noinspection PyDefaultArgument
class DB(ABC):
    @abstractmethod
    def query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any] = {}) -> list[list[tuple[Any, float]]]:
        """
        Perform an approximate nearest neighbor (ANN) search of the ColBERT embeddings.
        This informs the decision of which documents to fetch for full ColBERT scoring.

        Args:
            embeddings: A 2D tensor of ColBERT embeddings to compare against.
            limit: The maximum number of results to return for each embedding.
            params: Additional search parameters, if any.

        Returns:
            A list of lists, one per embedding, where each inner list contains tuples of (PK, similarity)
            for the chunks closest to each query embedding.
        """

    @abstractmethod
    def query_chunks(self, pks: list[Any]) -> Iterable[torch.Tensor]:
        """
        Retrieve all ColBERT embeddings for specific chunks so that ColBERT scores
        can be computed.

        Args:
            pks: A list of primary keys (of any object type) identifying the chunks.
            params: Additional search parameters, if any.

        Returns:
            An iterable of 2D tensors representing the ColBERT embeddings for each of the specified chunks.
        """
