import uuid
from typing import List, Any

import torch

from colbert_live.db.astra import AstraCQL
from cassandra.cluster import ResultSet


class CmdlineDB(AstraCQL):
    def __init__(self, keyspace: str, model_path: str, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, model_path, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # Create documents table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.documents (
                id uuid PRIMARY KEY,
                title text
            )
        """))

        # Create chunks table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.chunks (
                doc_id uuid,
                id int,
                body text,
                PRIMARY KEY (doc_id, id)
            )
        """))

        # Create chunk_embeddings table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.chunk_embeddings (
                doc_id uuid,
                chunk_id int,
                embedding_id int,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (doc_id, chunk_id, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        index_future = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.chunk_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)

        # Prepare statements
        self.insert_document_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.documents (id, title) VALUES (?, ?)
        """)
        self.insert_chunk_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.chunks (doc_id, id, body) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.chunk_embeddings (doc_id, chunk_id, embedding_id, embedding) VALUES (?, ?, ?, ?)
        """)
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT doc_id, chunk_id, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.chunk_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.chunk_embeddings WHERE doc_id = ? AND chunk_id = ?
        """)

        index_future.result()
        print("Schema ready")

    def add_record(self, title: str, chunks: List[str]):
        doc_id = uuid.uuid4()
        self.session.execute(self.insert_document_stmt, (doc_id, title))

        for i, chunk in enumerate(chunks):
            self.session.execute(self.insert_chunk_stmt, (doc_id, i, chunk))

        return doc_id

    def add_embeddings(self, doc_id: uuid.UUID, chunk_embeddings: List[torch.Tensor]):
        for chunk_id, embeddings in enumerate(chunk_embeddings):
            for embedding_id, embedding in enumerate(embeddings):
                self.session.execute(self.insert_embedding_stmt,
                                     (doc_id, chunk_id, embedding_id, embedding.tolist()))

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.doc_id, row.chunk_id), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]
