import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Dict, Tuple, List
from collections import defaultdict

from tqdm import tqdm
from more_itertools import chunked

from colbert_live import ColbertLive
from colbert_live.db.astra import execute_concurrent_async
from colbert_live.models import ColbertModel
from .db import BenchDB

LOTTE_DATA_PATH = os.path.expanduser("~/datasets/lotte")

def load_dataset(dataset: str, split: str) -> Tuple[Dict, Dict, Dict]:
    print(f"Loading dataset {dataset} ({split} split)...")
    data_path = os.path.join(LOTTE_DATA_PATH, dataset, split)
    
    # Load collection
    corpus = {}
    with open(os.path.join(data_path, "collection.tsv"), "r") as f:
        for line in f:
            pid, text = line.strip().split("\t")
            corpus[pid] = {"text": text}

    # Load queries and qrels for both search and forum
    queries = {}
    qrels = defaultdict(dict)
    for query_type in ["search", "forum"]:
        with open(os.path.join(data_path, f"questions.{query_type}.tsv"), "r") as f:
            for line in f:
                qid, text = line.strip().split("\t")
                queries[f"{query_type}_{qid}"] = text

        with open(os.path.join(data_path, f"qas.{query_type}.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                qid = f"{query_type}_{data['qid']}"
                for pid in data['answer_pids']:
                    qrels[qid][pid] = 1

    print(f"Dataset loaded. Corpus size: {len(corpus)}, Queries: {len(queries)}, Relevance judgments: {len(qrels)}")
    return corpus, queries, qrels

def process_document_batch(batch: List[Tuple[str, Dict]], db, colbert_live):
    chunk_raw_data = []
    chunk_texts = []
    insert_data = []

    for chunk_id, chunk in batch:
        content = chunk['text']
        chunk_texts.append(content)
        chunk_raw_data.append((chunk_id, "", content))
    
    embeddings = colbert_live.encode_chunks(chunk_texts)
    for (chunk_id, _, _), embedding_tensor in zip(chunk_raw_data, embeddings):
        insert_data.extend((chunk_id, i, e) for i, e in enumerate(embedding_tensor))

    chunk_stmt_with_parms = list(zip(cycle([db.insert_chunk_stmt]), chunk_raw_data))
    embeddings_stmt_with_parms = list(zip(cycle([db.insert_embeddings_stmt]), insert_data))
    return execute_concurrent_async(db.session, chunk_stmt_with_parms + embeddings_stmt_with_parms)

def is_populated(db):
    result = db.session.execute(f"SELECT * FROM {db.keyspace}.chunks LIMIT 1")
    return result.one() is not None

def compute_and_store_embeddings(corpus: dict, db, colbert_live):
    if is_populated(db):
        print("The chunks table is not empty. Skipping encoding and insertion.")
        return

    print("Encoding and inserting documents...")
    start_time = time.time()
    
    batch_size = 32
    insert_future = None
    for doc_batch in tqdm(chunked(corpus.items(), batch_size), total=len(corpus)//batch_size + 1, desc="Encoding and inserting"):
        next_insert_future = process_document_batch(doc_batch, db, colbert_live)
        if insert_future:
            insert_future.result()
        insert_future = next_insert_future

    end_time = time.time()
    print(f"Encoding and insertion completed. Time taken: {end_time - start_time:.2f} seconds")

def search_and_benchmark(queries: dict, n_ann_docs: int, n_colbert_candidates: int, colbert_live: ColbertLive) -> Dict[str, Dict[str, float]]:
    def search(query_item: Tuple[str, str]) -> Tuple[str, Dict[str, float]]:
        query_id, query = query_item
        return (query_id, dict(colbert_live.search(query, n_colbert_candidates, n_ann_docs, n_colbert_candidates)))

    start_time = time.time()
    num_threads = 8
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = dict(tqdm(executor.map(search, queries.items()), total=len(queries), desc="Retrieving"))
    end_time = time.time()

    print(f"  Time: {end_time - start_time:.2f} seconds = {len(queries) / (end_time - start_time):.2f} QPS")
    return results

def write_rankings(results: Dict[str, Dict[str, float]], output_file: str):
    with open(output_file, "w") as f:
        for qid, doc_scores in results.items():
            for rank, (pid, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True), start=1):
                qid_bare = qid.split("_")[1]
                f.write(f"{qid_bare}\t{pid}\t{rank}\t{score}\n")

def evaluate_lotte(dataset: str, split: str, query_type: str):
    model_name = 'answerdotai/answerai-colbert-small-v1'
    doc_pool = 2
    n_ann_docs = 240
    n_maxsim_candidates = 15
    tokens_per_query = 32

    ks_name = f"lotte_{dataset.replace('-', '')}_colbert_{doc_pool}"

    model = ColbertModel(model_name, tokens_per_query=tokens_per_query)
    db = BenchDB(ks_name, model.dim, os.environ.get('ASTRA_DB_ID'), os.environ.get('ASTRA_DB_TOKEN'))
    colbert_live = ColbertLive(db, model, doc_pool_factor=doc_pool)

    corpus, all_queries, all_qrels = load_dataset(dataset, split)
    compute_and_store_embeddings(corpus, db, colbert_live)

    queries = {qid: query for qid, query in all_queries.items() if qid.startswith(f"{query_type}_")}

    print(f"Evaluating {dataset} ({query_type} queries) @ {tokens_per_query} TPQ")
    results = search_and_benchmark(queries, n_ann_docs, n_maxsim_candidates, colbert_live)

    output_dir = f"lotte_rankings/{split}]/colbert"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}.{query_type}.ranking.tsv")
    write_rankings(results, output_file)

    print(f"Rankings written to {output_file}")

def main(datasets):
    for dataset in datasets:
        for query_type in ["search", "forum"]:
            evaluate_lotte(dataset, "test", query_type)

if __name__ == "__main__":
    all_datasets = ["writing", "recreation", "science", "technology", "lifestyle"]

    if len(sys.argv) > 1:
        datasets_to_run = [d for d in sys.argv[1:] if d in all_datasets]
        unrecognized_datasets = [d for d in sys.argv[1:] if d not in all_datasets]
        if unrecognized_datasets:
            print(f"Skipping unrecognized datasets: {', '.join(unrecognized_datasets)}")
    else:
        datasets_to_run = all_datasets

    print(f"Evaluating datasets: {', '.join(datasets_to_run)}")
    main(datasets_to_run)
