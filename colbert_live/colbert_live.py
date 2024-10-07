import math
from typing import Any, Optional

import numpy as np
import torch
from PIL.Image import Image
from colbert.modeling.checkpoint import pool_embeddings_hierarchical
from sklearn.cluster import AgglomerativeClustering

from .models import Model
from .db import DB


def _expand(x, a, b, c):
    """
    increases x by a factor >= 1 that decays as x increases
    """
    if x < 1:
        return 0
    return max(x, int(a + b*x + c*x*math.log(x)))

def _pool_query_embeddings(query_embeddings: torch.Tensor, max_distance: float) -> torch.Tensor:
    # Convert embeddings to numpy for clustering
    embeddings_np = query_embeddings.cpu().numpy()
    # Cluster
    clustering = AgglomerativeClustering(
        metric='cosine',
        linkage='average',
        distance_threshold=max_distance,
        n_clusters=None
    )
    labels = clustering.fit_predict(embeddings_np)

    # Pool the embeddings based on cluster assignments
    pooled_embeddings = []
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_embeddings = query_embeddings[cluster_indices]
        if len(cluster_embeddings) > 1:
            # average the embeddings in the cluster
            pooled_embedding = cluster_embeddings.mean(dim=0).to(query_embeddings.device)
            # re-normalize the pooled embedding
            pooled_embedding = pooled_embedding / torch.norm(pooled_embedding, p=2)
            pooled_embeddings.append(pooled_embedding)
        else:
            # only one embedding in the cluster, no need to do extra computation
            pooled_embeddings.append(cluster_embeddings[0])

    return torch.stack(pooled_embeddings)


class ColbertLive:
    def __init__(self,
                 db: DB,
                 model: Model,
                 doc_pool_factor: int = 2,
                 query_pool_distance: float = 0.03
                 ):
        """
        Initialize the ColbertLive instance.

        Args:
            db: The database instance to use for querying and storing embeddings.
            Model: The Model instance to use for encoding queries and documents.  ColbertModel and ColpaliModel
            are the two implementations provided by colbert-live.
            doc_pool_factor (optional): The factor by which to pool document embeddings, as the number of embeddings per cluster.
                `None` to disable.
            query_pool_distance (optional): The maximum cosine distance across which to pool query embeddings.
                `0.0` to disable.

            doc_pool_factor is only used by encode_chunks.

            query_pool_distance and tokens_per_query are only used by search and encode_query.
        """
        self.db = db
        self.model = model
        self.doc_pool_factor = doc_pool_factor
        self.query_pool_distance = query_pool_distance

    def encode_query(self, q: str) -> torch.Tensor:
        """
        Encode a query string into a tensor of embeddings.  Called automatically by search,
        but also exposed here as a public method.

        Args:
            q: The query string to encode.

        Returns:
            A 2D tensor of query embeddings.
        """
        query_embeddings = self.model.encode_query(q)  # Get embeddings for a single query
        query_embeddings = query_embeddings.float()  # Convert to float32
        if not self.query_pool_distance:
            result = query_embeddings  # Add batch dimension
        else:
            result = _pool_query_embeddings(query_embeddings, self.query_pool_distance)  # Add batch dimension
        return result

    def encode_chunks(self, chunks: list[str | Image]) -> list[torch.Tensor]:
        """
        Encode a batch of document chunks into tensors of embeddings.

        Args:
            chunks: A list of content strings or images to encode.  (The type of data must match what
            your Model can process.)

        Performance note: while it is perfectly legitimate to encode a single chunk at a time, this method
        is designed to support multiple chunks because that means we can dispatch all of that work to the GPU
        at once.  The overhead of invoking a CUDA kernel is *very* significant, so for an initial bulk load
        it is much faster to encode in larger batches.  (OTOH, if you are encoding without the benefit of
        GPU acceleration, then this should not matter very much.)

        Returns:
            A list of 2D tensors of float32 embeddings, one for each input chunk.
            Each tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).
        """
        embeddings_list = [Di.float() for Di in self.model.encode_doc(chunks)]

        # Apply pooling if pool_factor > 1
        if self.doc_pool_factor and self.doc_pool_factor > 1:
            for i, Di in enumerate(embeddings_list):
                # Convert to float32 before pooling
                Di, _ = pool_embeddings_hierarchical(
                    Di,
                    [Di.shape[0]],  # Single document length
                    pool_factor=self.doc_pool_factor,
                    protected_tokens=0
                )
                embeddings_list[i] = Di

        return embeddings_list

    def _load_data_and_construct_tensors(self, chunk_ids: list[Any]) -> list[torch.Tensor]:
        all_embeddings = []

        results = self.db.query_chunks(chunk_ids)
        for embeddings_for_chunk in results:
            packed_one_chunk = self.model.to_device(embeddings_for_chunk)
            all_embeddings.append(packed_one_chunk)

        return all_embeddings

    MAX_LIMIT = 1000

    def search(self,
               query: str,
               k: int = 10,
               n_ann_docs: Optional[int] = None,
               n_maxsim_candidates: Optional[int] = None
               ) -> list[tuple[Any, float]]:
        """
        Perform a ColBERT search and return the top chunk IDs with their scores.

        Args:
            query: The query string to search for.
            k: The number of top chunks to return.
            n_ann_docs: The number of chunks to retrieve for each embedding in the initial ANN search.
            n_maxsim_candidates: The number of top candidates to consider for full ColBERT scoring
            after combine the results of the ANN searches.

            If n_ann_docs and/or n_colbert_candidates are not specified, a best guess will be derived
            from top_k.

        Performance note: search is `O(log n_ann_docs) + O(n_colbert_candidates)`.  (And O(tokens_per_query),
        if your queries are sufficiently long).  Thus, you can generally afford to overestimate `n_ann_docs`,
        but you will want to keep `n_colbert_candidates` as low as possible.

        Returns:
            list[tuple[Any, float]]: A list of tuples of (chunk_id, ColBERT score) for the top k chunks.
        """
        Q = self.encode_query(query)
        return self._search(Q, k, n_ann_docs, n_maxsim_candidates)

    def _search(self, query_encodings, k, n_ann_docs, n_maxsim_candidates):
        """
        Search with precomputed query embeddings.
        Exposed for vidore-benchmark, which wants to batch-compute query embeddings up front
        """
        if n_ann_docs is None:
            # f(1) = 105, f(10) = 171, f(100) = 514, f(500) = 998
            n_ann_docs = _expand(k, 94.9, 11.0, -1.48)
        if n_maxsim_candidates is None:
            # f(1) = 9, f(10) = 20, f(100) = 119, f(900) = 1000
            n_maxsim_candidates = _expand(k, 8.82, 1.13, -0.00471)
        # compute the max score for each term for each doc
        chunks_per_query = {}
        for n, rows in enumerate(self.db.query_ann(query_encodings, n_ann_docs)):
            for chunk_id, similarity in rows:
                key = (chunk_id, n)
                chunks_per_query[key] = max(chunks_per_query.get(key, -1), similarity)
        if not chunks_per_query:
            return []  # empty database
        # sum the partial scores and identify the top candidates
        chunks = {}
        for (chunk_id, qv), similarity in chunks_per_query.items():
            chunks[chunk_id] = chunks.get(chunk_id, 0) + similarity
        candidates = sorted(chunks, key=chunks.get, reverse=True)[:n_maxsim_candidates]
        # Load document encodings
        doc_encodings = self._load_data_and_construct_tensors(candidates)
        # Calculate full ColBERT scores
        scores = self.model.score(query_encodings, doc_encodings)
        # Map the scores back to chunk IDs and sort
        results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        # Convert tensor scores to Python floats and return top k results
        return [(chunk_id, score.item()) for chunk_id, score in results[:k]]
