import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from colbert_live.colbert_live import ColbertLive
from colbert_live.models import Model
from .db import CmdlineDB
from pdf2image import convert_from_path
from PIL import Image
import io

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

        page_pngs = []
        for image in page_images:
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                page_pngs.append(output.getvalue())

        doc_id = db.add_documents(page_pngs)  # Create a new document ID
        for image in tqdm(page_images, desc="Encoding pages"):
            page_embeddings = colbert_live.encode_chunks([image])
            db.add_embeddings(doc_id, page_embeddings)

        print(f"Document '{filename}' added with ID: {doc_id}")


def search_documents(db, colbert_live, query, k=5):
    results = colbert_live.search(query, k=k)
    print("\nSearch results:")
    print("Score  Page  Document ID")
    for i, (chunk_pk, score) in enumerate(results, 1):
        doc_id, page_num = chunk_pk
        print(f"{score:.3f}  {page_num}    {doc_id}")

    if results:
        top_doc_id, top_page_num = results[0][0]
        page_content = db.get_page_content(top_doc_id, top_page_num)
        print(f"\nMost relevant page (Document ID: {top_doc_id}, Page: {top_page_num}):")

        if page_content:
            print(f"Page content size: {len(page_content)} bytes")
            try:
                image = Image.open(io.BytesIO(page_content))
                print(f"Image dimensions: {image.size[0]}x{image.size[1]}")
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    image.save(temp_file.name, format="PNG")
                    print(f"Image saved to: {temp_file.name}")
                    print("You can open this file to view the image.")
            except Exception as e:
                print(f"Error creating image: {e}")
        else:
            print("Error: Page content is empty")
    else:
        print("\nNo results found.")


def main():
    parser = argparse.ArgumentParser(description="Colbert Live Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add new documents")
    add_parser.add_argument("filenames", nargs="+", help="Filenames of documents to add")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    model = Model.from_name_or_path('vidore/colpali-v1.2')
    db = CmdlineDB('colpali',
                   model.dim,
                   os.environ.get('ASTRA_DB_ID'),
                   os.environ.get('ASTRA_DB_TOKEN'))
    colbert_live = ColbertLive(db, model)

    if args.command == "add":
        add_documents(db, colbert_live, args.filenames)
    elif args.command == "search":
        search_documents(db, colbert_live, args.query, args.k)


if __name__ == "__main__":
    main()
