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
