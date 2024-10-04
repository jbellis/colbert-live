# Colbert Live!

This documents how to use Colbert Live with the AstraCQL DB implementation.  First, here's the ColbertLive class itself:
## ColbertLive

### Constructor

#### `__init__(self, db: colbert_live.db.db.DB, model: colbert_live.models.Model, doc_pool_factor: int = 2, query_pool_distance: float = 0.03)`

Initialize the ColbertLive instance.


**Arguments:**

- `db`: The database instance to use for querying and storing embeddings.
- `Model`: The Model instance to use for encoding queries and documents.  ColbertModel and ColpaliModel
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

#### `search(self, query: str, k: int = 10, n_ann_docs: Optional[int] = None, n_maxsim_candidates: Optional[int] = None) -> list[tuple[typing.Any, float]]`

Perform a ColBERT search and return the top chunk IDs with their scores.


**Arguments:**

- `query`: The query string to search for.
- `k`: The number of top chunks to return.
- `n_ann_docs`: The number of chunks to retrieve for each embedding in the initial ANN search.
- `n_maxsim_candidates`: The number of top candidates to consider for full ColBERT scoring
  after combine the results of the ANN searches.
  
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
from .db import CmdlineDB


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

def add_documents(db, colbert_live, filenames):
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

        doc_id = db.add_record(pngs, all_embeddings)
        print(f"Document '{filename}' {len(pngs)} pages added with ID {doc_id}")


def search_documents(db, colbert_live, query, k=5):
    results = colbert_live.search(query, k=k)
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add new documents")
    add_parser.add_argument("filenames", nargs="+", help="Filenames of documents to add")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    model = ColpaliModel()
    db = CmdlineDB('colpali', model.dim, os.getenv("ASTRA_DB_ID"), os.getenv("ASTRA_DB_TOKEN"))
    colbert_live = ColbertLive(db, model)

    if args.command == "add":
        add_documents(db, colbert_live, args.filenames)
    elif args.command == "search":
        search_documents(db, colbert_live, args.query, args.k)


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
- `query_ann_stmt`: The prepared statement for ANN queries.
- `query_chunks_stmt`: The prepared statement for chunk queries.
  
- `Subclasses must implement`: 
- `- prepare`: Set up necessary database statements and perform any required table manipulation.
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
  
- `Raises`: 
- `Exception`: If Astra credentials are incomplete or connection fails.

### Constructor

#### `__init__(self, keyspace: str, embedding_dim: int, astra_db_id: str | None, astra_token: str | None, verbose: bool = False)`

### Methods

#### `prepare(self, embedding_dim: int)`

Prepare the database schema and query statements.  AstraCQL creates the keyspace if necessary;
everything else is up to you.

This method should be implemented by subclasses to set up the necessary
database structure and prepare statements for querying.


**Arguments:**

- `embedding_dim (int)`: The dimension of the ColBERT embeddings.
  
- `Expected implementations`: 
  1. Create required tables (if not exists)
  2. Create necessary indexes (if not exists)
- `3. Prepare two main statements`: 
- `a) query_ann_stmt`: For approximate nearest neighbor search
- `- Parameters`: [query_embedding, query_embedding, limit]
- `- Expected result`: [(primary_key, similarity)]
- `Example`: 
  SELECT pk, similarity_cosine(embedding, ?) AS similarity
  FROM table
  ORDER BY embedding ANN OF ?
  LIMIT ?
  
- `b) query_chunks_stmt`: For retrieving embeddings by primary key
- `- Parameters`: [primary_key]
- `- Expected result`: [embedding]
- `Example`: 
  SELECT embedding
  FROM table
  WHERE pk = ?
  
- `Note`: 
  - Ensure that compound primary keys are represented as tuples in the results.
  - The results of these queries will be processed by process_ann_rows and process_chunk_rows, respectively.

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

Note:
- Ensure that the returned tensors match the expected embedding dimension.
- If your database stores embeddings in a different format, convert them to torch.Tensor here.

#### `query_ann(self, embeddings: torch.Tensor, limit: int) -> list[list[tuple[typing.Any, float]]]`

#### `query_chunks(self, chunk_ids: list[typing.Any]) -> list[torch.Tensor]`



```

## AstraCQL example
```
import uuid
from typing import List, Any

import torch
from cassandra.concurrent import execute_concurrent_with_args

from colbert_live.db.astra import AstraCQL
from cassandra.cluster import ResultSet


class CmdlineDB(AstraCQL):
    def __init__(self, keyspace: str, embedding_dim: int, astra_db_id: str, astra_token: str):
        super().__init__(keyspace, embedding_dim, astra_db_id, astra_token, verbose=True)

    def prepare(self, embedding_dim: int):
        # Create tables asynchronously
        futures = []

        # for simplicity we don't actually have a records table, but if we
        # wanted to add things like title, creation date, etc., that's where it would go

        # Create chunks table
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
                embedding vector<float, {embedding_dim}>,
                PRIMARY KEY (record_id, page_num, embedding_id)
            )
        """))

        # Wait for all CREATE TABLE operations to complete
        for future in futures:
            future.result()

        # Create colbert_ann index
        index_future = self.session.execute_async(f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann 
            ON {self.keyspace}.page_embeddings(embedding) 
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{ 'source_model': 'bert' }}
        """)

        # Prepare statements
        self.insert_page_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.pages (record_id, num, body) VALUES (?, ?, ?)
        """)
        self.insert_embedding_stmt = self.session.prepare(f"""
            INSERT INTO {self.keyspace}.page_embeddings (record_id, page_num, embedding_id, embedding) VALUES (?, ?, ?, ?)
        """)

        index_future.result()
        self.query_ann_stmt = self.session.prepare(f"""
            SELECT record_id, page_num, similarity_cosine(embedding, ?) AS similarity
            FROM {self.keyspace}.page_embeddings
            ORDER BY embedding ANN OF ?
            LIMIT ?
        """)
        self.query_chunks_stmt = self.session.prepare(f"""
            SELECT embedding FROM {self.keyspace}.page_embeddings WHERE record_id = ? AND page_num = ?
        """)

        print("Schema ready")

    def add_record(self, pages: list[bytes], embeddings: list[torch.Tensor]):
        record_id = uuid.uuid4()
        L = [(record_id, num, body) for num, body in enumerate(pages, start=1)]
        execute_concurrent_with_args(self.session, self.insert_page_stmt, L)

        L = [(record_id, page_num, embedding_id, embedding)
             for page_num in range(1, len(embeddings) + 1)
             for embedding_id, embedding in enumerate(embeddings[page_num - 1])]
        execute_concurrent_with_args(self.session, self.insert_embedding_stmt, L)

        return record_id

    def process_ann_rows(self, result: ResultSet) -> List[tuple[Any, float]]:
        return [((row.record_id, row.page_num), row.similarity) for row in result]

    def process_chunk_rows(self, result: ResultSet) -> List[torch.Tensor]:
        return [torch.tensor(row.embedding) for row in result]

    def get_page_body(self, chunk_pk: tuple) -> bytes:
        record_id, page_num = chunk_pk
        query = f"SELECT body FROM {self.keyspace}.pages WHERE record_id = %s AND num = %s"
        result = self.session.execute(query, (record_id, page_num))
        return result.one()[0]

```
