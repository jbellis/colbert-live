import argparse
import os

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model
from db import CmdlineDB


def add_document(db, colbert_live, title, chunks):
    doc_id = db.add_document(title, chunks)
    chunk_embeddings = colbert_live.encode_chunks(chunks)
    db.add_embeddings(doc_id, chunk_embeddings)
    print(f"Document added with ID: {doc_id}")


def search_documents(db, colbert_live, query, k=5):
    results = colbert_live.search(query, k=k)
    print("\nSearch results:")
    print("Score  Chunk  Title")
    for i, (chunk_pk, score) in enumerate(results, 1):
        doc_id, chunk_id = chunk_pk
        rows = db.session.execute(f"SELECT title FROM {db.keyspace}.documents WHERE id = %s", [doc_id])
        title = rows.one().title
        print(f"{score:.3f}  {chunk_id}      {title}")


def main():
    parser = argparse.ArgumentParser(description="Colbert Live Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a new document")
    add_parser.add_argument("title", help="Document title")
    add_parser.add_argument("chunks", nargs="+", help="Document chunks")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    model = Model.from_name_or_path('answerdotai/answerai-colbert-small-v1')
    db = CmdlineDB('colbertlive',
                   model.dim,
                   os.environ.get('ASTRA_DB_ID'),
                   os.environ.get('ASTRA_DB_TOKEN'))
    colbert_live = ColbertLive(db, model)

    if args.command == "add":
        add_document(db, colbert_live, args.title, args.chunks)
    elif args.command == "search":
        search_documents(db, colbert_live, args.query, args.k)


if __name__ == "__main__":
    main()
