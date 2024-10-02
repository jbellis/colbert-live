# ColBERT Live!

ColBERT Live! implements efficient [ColBERT](https://github.com/stanford-futuredata/ColBERT) search on top of vector indexes that support live updates 
(without rebuilding the entire index) as well as arbitrary predicates against other indexed fields.

## Background

ColBERT (Contextualized Late Interaction over BERT) is a state-of-the-art semantic search model that combines the effectiveness of BERT-based language models with the performance required for practical, large-scale search applications.

Compared to traditional dense passage retrieval (i.e. vector-per-passage) ColBERT is particularly strong at handling unusual terms and short queries.

It is reasonable to think of ColBERT as combining the best of semantic vector search with traditional keyword search a la BM25, but without having
to tune the weighting of hybrid search or dealing with corner cases where the vector and keyword sides play poorly together. 

However, the initial ColBERT implementation is designed around a custom index that cannot be updated incrementally,
and can only be combined with other indexes with difficulty. Adding, modifying, or removing documents 
from the custom index requires reindexing the entire collection, which can be prohibitively slow for large datasets.

## ColBERT Live!

ColBERT Live! implements ColBERT on any vector database. This means you can add, modify, or remove documents 
from your search system without the need for costly reindexing of the entire collection, making it ideal for dynamic content environments.
It also means that you can easily apply other predicates such as access controls or metadata filters from your database to your vector searches.
ColBERT Live! features

- Efficient ColBERT search implementation
- Support for live updates to the vector index
- Abstraction layer for database backends, starting with [AstraDB](https://www.datastax.com/products/astra)
- State of the art ColBERT techniques including:
  - Answer.AI ColBERT model for higher relevance
  - Document embedding pooling for reduced storage requirements
  - Query embedding pooling for improved search performance

## Installation

You can install ColBERT Live! using pip:

```bash
pip install colbert-live
```

## Usage

- Subclass your database backend and implement the required methods for retrieving embeddings:
  ```
  from colbert_live.db.astra import AstraCql
  
  class MyDB(AstraCql):
    ...
  
  db = MyDB()
  ```
- Instantiate 
  `model = colbert_live.models.ColbertModel()` or
  `model = colbert_live.models.ColpaliModel()`
- Initialize `colbert = ColbertLive(db, model)`
- Call `colbert.search(query_str, top_k)`

Two cheat sheets are available:
- Using ColBERT Live! with Astra:
  [for humans](https://github.com/jbellis/colbert-live/blob/master/docs/colbert-live-astra.md);
  [for LLMs](https://raw.githubusercontent.com/jbellis/colbert-live/refs/heads/master/docs/colbert-live-astra.md)
- Implementing a new DB subclass:
  [for humans](https://github.com/jbellis/colbert-live/blob/master/docs/colbert-live-db.md);
  [for LLMs](https://raw.githubusercontent.com/jbellis/colbert-live/refs/heads/master/docs/colbert-live-db.md)

## Supported databases

ColBERT Live! initially supports [DataStax Astra](https://www.datastax.com/products/astra) out of the box.
Adding support for other databases is straightforward; check out 
[the Astra implementation](https://github.com/jbellis/colbert-live/blob/master/colbert_live/db/astra.py) 
for an example to follow.  If you're not concerned about making it reusable, you just have to implement
[the two methods of the base DB class](https://github.com/jbellis/colbert-live/blob/master/colbert_live/db/db.py).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.txt) file for details.
