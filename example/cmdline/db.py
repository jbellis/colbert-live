import uuid
from typing import List, Any, Tuple

import torch
from cassandra.concurrent import execute_concurrent_with_args
from cassandra.query import PreparedStatement

from colbert_live.db.astra import AstraCQL
from cassandra.cluster import ResultSet

from colbert_live.db.sqlite import Sqlite3DB
import sqlite3
import numpy as np


class CmdlineAstraDB(AstraCQL):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # for simplicity, we don't actually have a records table, but if we
        # wanted to add things like title, creation date, etc., that's where it would go

        # Create pages table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.pages (
                record_id uuid,
                num int,
                body blob,
                PRIMARY KEY (record_id, num)
            )
        """))

        # Create page_embeddings table
        futures.append(self.session.execute_async(f"""
            CREATE TABLE IF NOT EXISTS {self.keyspace}.page_embeddings (
                record_id uuid,
                page_num int,
                embedding_id int,
                tags set<text>,
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (record_id, page_num, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        i1 = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.page_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)
        i2 = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_tags 
            ON {self.keyspace}.page_embeddings(tags) 
            USING 'StorageAttachedIndex'
        """)
        index_futures = [i1, i2]

        # Prepare statements
        self.insert_page_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.pages (record_id, num, body) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.page_embeddings (record_id, page_num, embedding_id, embedding, tags) VALUES (?, ?, ?, ?, ?)
        """)

        [index_future.result() for index_future in index_futures]
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT record_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_ann_with_tag_stmt = self.session.prepare(f"""
            SELECT record_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            WHERE tags CONTAINS ?
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.page_embeddings WHERE record_id = ? AND page_num = ?
        """)

        print("Schema ready")

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor], tags: set[str] = set()):
        record_id = uuid.uuid4()
        L = [(record_id, num, body) for num, body in enumerate(pages, start=1)]
        execute_concurrent_with_args(self.session, self.insert_page_stmt, L)

        L = [(record_id, page_num, embedding_id, embedding, tags)
             for page_num in range(1, len(embeddings) + 1)
             for embedding_id, embedding in enumerate(embeddings[page_num - 1])]
        execute_concurrent_with_args(self.session, self.insert_embedding_stmt, L)

        return record_id

    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> tuple[PreparedStatement, list[tuple]]:
        tag = params.get('tag')
        if tag:
            params_list = [(emb, tag, emb, limit) for emb in embeddings.tolist()]
            return self.query_ann_with_tag_stmt, params_list
        else:
            params_list = [(emb, emb, limit) for emb in embeddings.tolist()]
            return self.query_ann_stmt, params_list

    def get_query_chunks_stmt(self) -> PreparedStatement:
        return self.query_chunks_stmt

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.record_id, row.page_num), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        query = f"SELECT body FROM {self.keyspace}.pages WHERE record_id = %s AND num = %s"
        result = self.session.execute(query, (record_id, page_num))
        return result.one()[0]


class CmdlineSqlite3DB(Sqlite3DB):
    def __init__(self, db_path: str, embedding_dim: int):
        super().__init__(db_path, embedding_dim, verbose=True)

    def prepare(self, embedding_dim: int):
        self.cursor.executescript(f"""
            CREATE TABLE IF NOT EXISTS pages (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                num INTEGER,
                body BLOB,
                UNIQUE (record_id, num)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS page_embedding_vectors USING vec0(
                embedding FLOAT[{embedding_dim}]
            );

            CREATE TABLE IF NOT EXISTS page_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER,
                page_num INTEGER,
                embedding_id INTEGER,
                page_embedding_vector_rowid INTEGER,
                FOREIGN KEY (record_id, page_num) REFERENCES pages(record_id, num),
                FOREIGN KEY (page_embedding_vector_rowid) REFERENCES page_embedding_vectors(rowid)
            );
        """)

        # insert statements
        self.insert_page_stmt = "INSERT INTO pages (num, body) VALUES (?, ?) RETURNING record_id"
        self.insert_embedding_stmt = """
            INSERT INTO page_embeddings (record_id, page_num, embedding_id, page_embedding_vector_rowid)
            VALUES (?, ?, ?, ?)
        """
        self.insert_embedding_vector_stmt = "INSERT INTO page_embedding_vectors (embedding) VALUES (vec_f32(?))"

        # queries
        self.query_ann_stmt = f"""
            SELECT record_id, page_num, distance
            FROM (
                SELECT rowid, distance
                FROM page_embedding_vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            ) AS pev
            JOIN page_embeddings pe ON pe.page_embedding_vector_rowid = pev.rowid
        """
        self.query_chunks_stmt = """
            SELECT pev.embedding
            FROM page_embeddings pe
            JOIN page_embedding_vectors pev ON pe.page_embedding_vector_rowid = pev.rowid
            WHERE pe.record_id = ? AND pe.page_num = ?
        """

        print("Schema ready")

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor]) -> int:
        assert pages
        for page_num, (page, page_embeddings) in enumerate(zip(pages, embeddings), start=1):
            # Insert page
            self.cursor.execute(self.insert_page_stmt, (page_num, page))
            record_id = self.cursor.fetchone()[0]

            # Insert embeddings for this page
            for embedding_id, embedding in enumerate(page_embeddings):
                # Insert the embedding vector
                self.cursor.execute(self.insert_embedding_vector_stmt, (embedding.cpu().numpy().tobytes(),))
                embedding_vector_rowid = self.cursor.lastrowid

                # Insert the embedding metadata
                self.cursor.execute(self.insert_embedding_stmt, 
                    (record_id, page_num, embedding_id, embedding_vector_rowid))

        self.conn.commit()
        return record_id

    def get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, Any]) -> Tuple[str, List[Any]]:
        params_list = [(emb.cpu().numpy().tobytes(), limit) for emb in embeddings]
        return self.query_ann_stmt, params_list

    def get_query_chunks_stmt(self) -> str:
        return self.query_chunks_stmt

    def process_ann_rows(self, result: List[sqlite3.Row]) -> List[Tuple[Any, float]]:
        return [((row[0], row[1]), row[2]) for row in result]

    def process_chunk_rows(self, result: List[sqlite3.Row]) -> List[torch.Tensor]:
        return [torch.from_numpy(np.frombuffer(row[0], dtype=np.float32)) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        self.cursor.execute("SELECT body FROM pages WHERE record_id = ? AND num = ?", (record_id, page_num))
        result = self.cursor.fetchone()
        return result[0]
