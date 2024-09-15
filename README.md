# ColBERT Live!

ColBERT Live! implements efficient [ColBERT](https://github.com/stanford-futuredata/ColBERT) search on top of vector indexes that support live updates (without rebuilding the entire index).

## Background

ColBERT (Contextualized Late Interaction over BERT) is a state-of-the-art semantic search model that combines the effectiveness of BERT-based language models with the performance required for practical, large-scale search applications.

Compared to traditional dense passage retrieval (i.e. vector-per-passage) ColBERT is particularly strong at handling unusual terms and short queries.

It's reasonable to think of ColBERT as combining the best of semantic vector search with traditional keyword search a la BM25, but without having
to tune the weighting of hybrid search or dealing with corner cases where the vector and keyword sides play poorly together. 

However, the initial ColBERT implementation is designed around a custom index that cannot be updated incrementally. This means that adding, modifying, or removing documents from the search system requires reindexing the entire collection, which can be prohibitively slow for large datasets. 

## ColBERT Live!

ColBERT Live! implements ColBERT on any vector index. This means you can add, modify, or remove documents from your search system without the need for costly reindexing of the entire collection, making it ideal for dynamic content environments.
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

- Subclass your database backend and implement the required methods for retrieving embeddings
- Initialize ColbertLive(db)
- Call ColbertLive.search(query_str, top_k)

Here's the code from the `cmdline` example, which implements adding and searching multi-chunk documents from the commandline. 

```python
class CmdlineDB(AstraDB):
    # see cmdline/db.py for details

def add_document(db, colbert_live, title, chunks):
    doc_id = db.add_document(title, chunks)
    chunk_embeddings = colbert_live.encode_chunks(chunks)
    db.add_embeddings(doc_id, chunk_embeddings)
    print(f"Document added with ID: {doc_id}")


def search_documents(db, colbert_live, query, k=5):
    results = colbert_live.search(query, k=k)
    print("\nSearch results:")
    for i, (chunk_pk, score) in enumerate(results, 1):
        doc_id, chunk_id = chunk_pk
        print(doc_id, type(doc_id))
        rows = db.session.execute(f"SELECT title FROM {db.keyspace}.documents WHERE id = %s", [doc_id])
        title = rows.one().title
        print(f"{i}. {title} (Score: {score:.4f})")


def main():
    args = ... # arg parsing skipped, see cmdline/main.py for details

    db = CmdlineDB('colbertlive',
                   'answerdotai/answerai-colbert-small-v1',
                   os.environ.get('ASTRA_DB_ID'),
                   os.environ.get('ASTRA_DB_TOKEN'))
    colbert_live = ColbertLive(db)

    if args.command == "add":
        add_document(db, colbert_live, args.title, args.chunks)
    elif args.command == "search":
        search_documents(db, colbert_live, args.query, args.k)
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.txt) file for details.
