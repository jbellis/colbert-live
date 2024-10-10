import os
import sqlite3
from abc import abstractmethod
from typing import Any, List, Tuple

import sqlite_vec
import torch

from .db import DB


class Sqlite3DB(DB):
    """
    Sqlite3DB implements the ColBERT Live DB interface for SQLite databases.

    This class provides a foundation for creating application-specific implementations.
    Subclasses should override the prepare, process_ann_rows, and process_chunk_rows methods
    to customize the behavior for their specific use case.

    Args:
        db_path (str): The path to the SQLite database file.
        embedding_dim (int): The dimension of the ColBERT embeddings.
        verbose (bool): If True, print verbose output.

    Attributes:
        conn: The database connection object.
        cursor: The database cursor object.

    Subclasses must implement:
    - prepare: Set up necessary database statements and perform any required table manipulation.
    - get_query_ann: Set up prepared statement and bind vars for ANN queries.
    - get_query_chunks_stmt: Return prepared statement for chunk queries.
    - process_ann_rows: Process the results of the ANN query.
    - process_chunk_rows: Process the results of the chunk query.
    See the docstrings of these methods for details.
    """

    def __init__(self,
                 db_path: str,
                 embedding_dim: int,
                 verbose: bool = False):
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)

        # load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec_dir = os.path.dirname(sqlite_vec.__file__)
        self.conn.load_extension(os.path.join(sqlite_vec_dir, 'vec0'))

        self.cursor = self.conn.cursor()

        self.prepare(embedding_dim)

    @abstractmethod
    def prepare(self, embedding_dim: int):
        """
        Prepare the database schema and query statements.

        This method should be implemented by subclasses to set up the necessary
        database structure and prepare any additional statements for querying.

        Args:
            embedding_dim (int): The dimension of the ColBERT embeddings.

        Expected implementations:
        1. Create required tables (if not exists)
        2. Create necessary indexes (if not exists)
        3. Prepare any additional statements needed for your specific implementation
        """

    @abstractmethod
    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Abstract method for setting up the ANN query.

        Args:
            embeddings: a 2D tensor of query embeddings to compare against.
            limit: The maximum number of results to return for each embedding.
            params: Additional parameters to pass to the query, if any.

        Returns:
            A tuple containing the SQL query string and a list of parameters to be used with the query.
        """

    @abstractmethod
    def get_query_chunks_stmt(self) -> str:
        """
        Abstract method for the chunks query.

        Returns:
            An SQL query string for retrieving embeddings by primary key.
        """

    @abstractmethod
    def process_ann_rows(self, result: List[sqlite3.Row]) -> List[Tuple[Any, float]]:
        """
        Process the result of the ANN query into a list of (primary_key, similarity) tuples.

        Args:
            result (List[sqlite3.Row]): The result set from the ANN query.

        Returns:
            List[Tuple[Any, float]]: A list of tuples, each containing a primary key and its similarity score.

        Example implementation:
            return [(row['primary_key'], row['similarity']) for row in result]

        Note:
        - The primary_key should match the structure used in your database schema.
        - For compound primary keys, return them as tuples, e.g., (doc_id, page_num).
        """

    @abstractmethod
    def process_chunk_rows(self, result: List[sqlite3.Row]) -> List[torch.Tensor]:
        """
        Process the result of the chunk query into a list of embedding tensors.

        Args:
            result (List[sqlite3.Row]): The result set from the chunk query.

        Returns:
            List[torch.Tensor]: A list of embedding tensors.

        Example implementation:
            return [torch.tensor(json.loads(row['embedding'])) for row in result]
        """

    # noinspection PyDefaultArgument
    def query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any] = {}) -> List[List[Tuple[Any, float]]]:
        if self.verbose:
            print(f'Querying ANN with {len(embeddings)} embeddings')
        
        query, query_params = self.get_query_ann(embeddings, limit, params)
        
        ann_results = []
        for emb, one_params in zip(embeddings, query_params):
            self.cursor.execute(query, one_params)
            result = self.cursor.fetchall()
            ann_results.append(self.process_ann_rows(result))

        return ann_results

    def query_chunks(self, chunk_ids: List[Any]) -> List[torch.Tensor]:
        if self.verbose:
            print(f'Loading embeddings from {len(chunk_ids)} chunks for full ColBERT scoring')
        
        query = self.get_query_chunks_stmt()

        chunk_results = []
        for chunk_id in chunk_ids:
            flattened_pk = chunk_id if isinstance(chunk_id, tuple) else (chunk_id,)
            self.cursor.execute(query, flattened_pk)
            result = self.cursor.fetchall()
            chunk_results.append(torch.stack(self.process_chunk_rows(result)))
        
        return chunk_results

    def __del__(self):
        self.conn.close()
