import sys
import re
import csv

def main():
    # Read the entire input from stdin
    data = sys.stdin.read()

    # Define a regex pattern to match each entry
    # The pattern captures:
    # 1. Dataset name
    # 2. TPQ value (unused)
    # 3. Keyspace name
    # 4. Query pool distance (unused)
    # 5. ann_docs
    # 6. maxsim_candidates
    # 7. Time in seconds (unused)
    # 8. QPS
    # 9. NDCG@1 (unused)
    # 10. NDCG@5 (unused)
    # 11. NDCG@10
    pattern = re.compile(
        r'([\w-]+) @ \d+ TPQ from keyspace (\w+), query pool distance [\d.]+, CL (\d+):(\d+)\s+'
        r'Time: [\d.]+ seconds = ([\d.]+) QPS\s+'
        r'NDCG@1: [\d.]+\s+'
        r'NDCG@5: [\d.]+\s+'
        r'NDCG@10: ([\d.]+)',
        re.MULTILINE
    )

    # Find all matches in the data
    matches = pattern.findall(data)

    # Initialize CSV writer to write to stdout
    writer = csv.writer(sys.stdout)
    
    # Write the header row
    writer.writerow(['dataset', 'doc_pooling', 'ann_docs', 'maxsim_candidates', 'qps', 'ndcg'])

    # Process each matched entry
    for match in matches:
        dataset = match[0]
        keyspace = match[1]
        ann_docs = match[2]
        maxsim_candidates = match[3]
        qps = match[4]
        ndcg10 = match[5]

        # Extract the last digit of the keyspace name for doc_pooling
        doc_pooling_match = re.search(r'(\d)', keyspace[::-1])
        doc_pooling = doc_pooling_match.group(1) if doc_pooling_match else ''

        # Write the extracted values as a row in the CSV
        writer.writerow([dataset, doc_pooling, ann_docs, maxsim_candidates, qps, ndcg10])

if __name__ == '__main__':
    main()

