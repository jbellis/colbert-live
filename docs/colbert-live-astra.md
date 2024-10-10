# Colbert Live!

This documents how to use Colbert Live with the AstraCQL DB implementation.  First, here's the ColbertLive class itself:
## ColbertLive

### Constructor

#### `__init__(self, db: colbert_live.db.db.DB, model: colbert_live.models.Model, doc_pool_factor: int = 2, query_pool_distance: float = 0.03)`

Initialize the ColbertLive instance.


**Arguments:**

- `db`: The database instance to use for querying and storing embeddings.
- `model`: The Model instance to use for encoding queries and documents.  ColbertModel and ColpaliModel
  are the two implementations provided by colbert-live.
- `doc_pool_factor (optional)`: The factor by which to pool document embeddings, as the number of embeddings per cluster.
  `None` to disable.
- `query_pool_distance (optional)`: The maximum cosine distance across which to pool query embeddings.
  `0.0` to disable.
  
  doc_pool_factor is only used by encode_chunks.
  
  query_pool_distance and tokens_per_query are only used by search and encode_query.

### Methods

#### `encode_chunks(self, chunks: list[str | PIL.Image.Image]) -> list[torch.Tensor]`

Encode a batch of document chunks into tensors of embeddings.


**Arguments:**

- `chunks`: A list of content strings or images to encode.  (The type of data must match what
  your Model can process.)
  
- `Performance note`: while it is perfectly legitimate to encode a single chunk at a time, this method
  is designed to support multiple chunks because that means we can dispatch all of that work to the GPU
  at once.  The overhead of invoking a CUDA kernel is *very* significant, so for an initial bulk load
  it is much faster to encode in larger batches.  (OTOH, if you are encoding without the benefit of
  GPU acceleration, then this should not matter very much.)
  

**Returns:**

  A list of 2D tensors of float32 embeddings, one for each input chunk.
  Each tensor has shape (num_embeddings, embedding_dim), where num_embeddings is variable (one per token).

#### `encode_query(self, q: str) -> torch.Tensor`

Encode a query string into a tensor of embeddings.  Called automatically by search,
but also exposed here as a public method.


**Arguments:**

- `q`: The query string to encode.
  

**Returns:**

  A 2D tensor of query embeddings.

#### `search(self, query: str, k: int = 10, n_ann_docs: Optional[int] = None, n_maxsim_candidates: Optional[int] = None, params: dict[str, typing.Any] = {}) -> list[tuple[typing.Any, float]]`

Perform a ColBERT search and return the top chunk IDs with their scores.


**Arguments:**

- `query`: The query string to search for.
- `k`: The number of top chunks to return.
- `n_ann_docs`: The number of chunks to retrieve for each embedding in the initial ANN search.
- `n_maxsim_candidates`: The number of top candidates to consider for full ColBERT scoring
  after combine the results of the ANN searches.
- `params`: Additional (non-vector) search parameters, if any.
  
  If n_ann_docs and/or n_colbert_candidates are not specified, a best guess will be derived
  from top_k.
  
- `Performance note`: search is `O(log n_ann_docs) + O(n_colbert_candidates)`.  (And O(tokens_per_query),
  if your queries are sufficiently long).  Thus, you can generally afford to overestimate `n_ann_docs`,
  but you will want to keep `n_colbert_candidates` as low as possible.
  

**Returns:**

  list[tuple[Any, float]]: A list of tuples of (chunk_id, ColBERT score) for the top k chunks.




## ColbertLive example
```
import argparse
import io
import os
import tempfile
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_path
from term_image.image import AutoImage

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model, ColpaliModel
from .db import CmdlineAstraDB, CmdlineSqlite3DB


def page_images_from(filename):
    file_path = Path(filename)
    if file_path.suffix.lower() != '.pdf':
        print(f"Warning: {filename} is not a PDF file. Skipping.")
        return None

    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(
            file_path,
            thread_count=os.cpu_count()-1,
            output_folder=path,
            paths_only=True
        )
        return [Image.open(image_path) for image_path in images]

def add_documents(db, colbert_live, filenames, tags: set[str], db_type: str):
    for filename in filenames:
        print(f"Extracting pages from '{filename}'...")
        page_images = page_images_from(filename)
        if page_images is None:
            continue

        pngs = []
        all_embeddings = []
        for i, image in enumerate(page_images):
            # Compute embeddings using the original image
            page_embeddings = colbert_live.encode_chunks([image])[0]
            all_embeddings.append(page_embeddings)

            # Resize the image to half its original resolution
            width, height = image.size
            resized_image = image.resize((width // 2, height // 2), Image.LANCZOS)

            with io.BytesIO() as output:
                resized_image.save(output, format="PNG")
                page_png = output.getvalue()
                pngs.append(page_png)

                # Display the resized page using term-image
                print(f"\nPage {i+1}:")
                term_image = AutoImage(resized_image)
                print(term_image)
        total_embeddings = sum(len(embeddings) for embeddings in all_embeddings)
        print(f'Inserting {total_embeddings} embeddings into the database...')
        if db_type == 'astra':
            doc_id = db.add_record(pngs, all_embeddings, tags)
        else:
            doc_id = db.add_record(pngs, all_embeddings)
        print(f"Document '{filename}' {len(pngs)} pages added with ID {doc_id}")
        if db_type == 'astra' and tags:
            print(f"  tags: {', '.join(tags)}")


def search_documents(db, colbert_live, query, k=5, tag=None):
    params = {'tag': tag} if tag else {}
    results = colbert_live.search(query, k=k, params=params)
    print("\nSearch results:")
    print("Score  Chunk  Title")
    for i, (chunk_pk, score) in enumerate(results[:3], 1):
        print(f"{i}. {score:.3f}  {chunk_pk}")
        page_body = db.get_page_body(chunk_pk)
        image = Image.open(io.BytesIO(page_body))
        image.show()
    if not results:
        print("No results found")


def main():
    parser = argparse.ArgumentParser(description="Colbert Live Demo")
    parser.add_argument("--db", choices=["astra", "sqlite3"], default="astra", help="Database type to use")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add new documents")
    add_parser.add_argument("filenames", nargs="+", help="Filenames of documents to add")
    add_parser.add_argument("--tags", help="Comma-separated list of tags to add to the documents")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--tag", help="Single tag to filter the search results")

    args = parser.parse_args()

    model = ColpaliModel()
    
    if args.db == "astra":
        db = CmdlineAstraDB('colpali', model.dim, os.getenv("ASTRA_DB_ID"), os.getenv("ASTRA_DB_TOKEN"))
    else:
        db = CmdlineSqlite3DB('colpali.db', model.dim)
    
    colbert_live = ColbertLive(db, model)

    if args.command == "add":
        if args.db == "sqlite3" and args.tags:
            print("Error: Tags are not supported with SQLite3 database.")
            return
        tags = set(s.strip() for s in args.tags.split(',')) if args.tags else set()
        add_documents(db, colbert_live, args.filenames, tags, args.db)
    elif args.command == "search":
        if args.db == "sqlite3" and args.tag:
            print("Error: Tag filtering is not supported with SQLite3 database.")
            return
        search_documents(db, colbert_live, args.query, args.k, args.tag)


if __name__ == "__main__":
    main()

```

Next, you will need to subclass AstraCQL:
```
## AstraCQL

AstraCQL implements the ColBERT Live DB interface for Astra CQL databases as well as local Cassandra.

This class provides a foundation for creating application-specific implementations.
Subclasses should override the prepare, process_ann_rows, and process_chunk_rows methods
to customize the behavior for their specific use case.


**Arguments:**

- `keyspace (str)`: The keyspace to use in the database. AstraCQL will create it if it doesn't exist.
- `embedding_dim (int)`: The dimension of the ColBERT embeddings.
- `astra_db_id (Optional[str])`: The Astra database ID (required for Astra connections).
- `astra_token (Optional[str])`: The Astra authentication token (required for Astra connections).
- `verbose (bool)`: If True, print verbose output.
  
- `Attributes`: 
- `session`: The database session object.
  
- `Subclasses must implement`: 
- `- prepare`: Set up necessary database statements and perform any required table manipulation.
- `- get_query_ann`: Set up prepared statement and bind vars for ANN queries.
- `- get_query_chunks_stmt`: Return prepared statement for chunk queries.
- `- process_ann_rows`: Process the results of the ANN query.
- `- process_chunk_rows`: Process the results of the chunk query.
  See the docstrings of these methods for details.
  
- `Example usage in a subclass`: 
- `class MyDB(AstraCQL)`: 
- `def prepare(self, embedding_dim)`: 
  # Create tables and indexes
  self.session.execute(f"CREATE TABLE IF NOT EXISTS ...")
  self.session.execute(f"CREATE CUSTOM INDEX IF NOT EXISTS ...")
  
  # Prepare statements
  self.query_ann_stmt = self.session.prepare(f"SELECT ... ORDER BY ... ANN OF ...")
  self.query_chunks_stmt = self.session.prepare(f"SELECT ... WHERE ...")
  
- `def process_ann_rows(self, result)`: 
  return [(row.primary_key, row.similarity) for row in result]
  
- `def process_chunk_rows(self, result)`: 
  return [torch.tensor(row.embedding) for row in result]
  
- `def get_query_ann(self, embeddings, limit, params)`: 
  params_list = [(emb, emb, limit) for emb in embeddings.tolist()]
  return self.query_ann_stmt, params_list
  
- `def get_query_chunks_stmt(self, chunk_ids)`: 
  return self.query_chunks_stmt
  
- `Raises`: 
- `Exception`: If Astra credentials are incomplete or connection fails.

### Constructor

#### `__init__(self, keyspace: str, embedding_dim: int, astra_db_id: str | None = None, astra_token: str | None = None, astra_endpoint: str | None = None, verbose: bool = False)`

### Methods

#### `get_query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, typing.Any]) -> tuple[cassandra.query.PreparedStatement, list[tuple]]`

Abstract method for setting up the ANN query.


**Arguments:**

- `embeddings`: a 2D tensor of query embeddings to compare against.
- `limit`: The maximum number of results to return for each embedding.
- `params`: Additional parameters to pass to the query, if any.
  

**Returns:**

  A prepared statement and a list of bind variables to pass to the query (one element per embedding).

#### `get_query_chunks_stmt(self) -> cassandra.query.PreparedStatement`

Abstract method for the chunks query.


**Returns:**

  A prepared statement for retrieving embeddings by primary key.

#### `prepare(self, embedding_dim: int)`

Prepare the database schema and query statements. AstraCQL creates the keyspace if necessary;
everything else is up to you.

This method should be implemented by subclasses to set up the necessary
database structure and prepare any additional statements for querying.


**Arguments:**

- `embedding_dim (int)`: The dimension of the ColBERT embeddings.
  
- `Expected implementations`: 
  1. Create required tables (if not exists)
  2. Create necessary indexes (if not exists)
  3. Prepare any additional statements needed for your specific implementation
  
- `Note`: 
  - The query_ann_stmt and query_chunks_stmt are now abstract methods and should be implemented separately.
  - Ensure that compound primary keys are represented as tuples in the results.

#### `process_ann_rows(self, result: cassandra.cluster.ResultSet) -> list[tuple[typing.Any, float]]`

Process the result of the ANN query into a list of (primary_key, similarity) tuples.


**Arguments:**

- `result (ResultSet)`: The result set from the ANN query.
  

**Returns:**

  List[Tuple[Any, float]]: A list of tuples, each containing a primary key and its similarity score.


Example implementation:
return [(row.primary_key, row.similarity) for row in result]

Note:
- The primary_key should match the structure used in your database schema.
- For compound primary keys, return them as tuples, e.g., (doc_id, page_num).

#### `process_chunk_rows(self, result: cassandra.cluster.ResultSet) -> list[torch.Tensor]`

Process the result of the chunk query into a list of embedding tensors.


**Arguments:**

- `result (ResultSet)`: The result set from the chunk query.
  

**Returns:**

  List[torch.Tensor]: A list of embedding tensors.


Example implementation:
return [torch.tensor(row.embedding) for row in result]

#### `query_ann(self, embeddings: torch.Tensor, limit: int, params: dict[str, typing.Any] = {}) -> list[list[tuple[typing.Any, float]]]`

#### `query_chunks(self, chunk_ids: list[typing.Any]) -> list[torch.Tensor]`



```

## Example of subclassing AstraCQL
```
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

```
