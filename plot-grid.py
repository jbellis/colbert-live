import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def is_pareto_optimal(point, points):
    for other in points:
        if all(other[i] >= point[i] for i in range(len(point))):
            if any(other[i] > point[i] for i in range(len(point))):
                return False
    return True

def main():
    # Read data from stdin
    data = defaultdict(list)
    header = True
    for line in sys.stdin:
        if header:
            header = False
            continue
        parts = line.strip().split(',')
        if len(parts) >= 6:
            dataset, doc_pool, ann_docs, maxsim_candidates, qps, ndcg = parts[:6]
            try:
                data[dataset].append((float(ndcg), float(qps), int(doc_pool), int(ann_docs), int(maxsim_candidates)))
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}", file=sys.stderr)

    # Plot setup
    plt.figure(figsize=(24, 16))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    for i, (dataset, points) in enumerate(data.items()):
        # Find Pareto-optimal points
        pareto_points = [p for p in points if is_pareto_optimal(p[:2], [x[:2] for x in points])]
        
        # Sort points by NDCG for line plotting
        pareto_points.sort(key=lambda x: x[0])
        
        # Extract values
        ndcg_values, qps_values, doc_pools, ann_docs_values, maxsim_candidates_values = zip(*pareto_points)
        
        # Plot scatter and line
        scatter = plt.scatter(ndcg_values, qps_values, label=dataset, color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], s=100)
        plt.plot(ndcg_values, qps_values, color=colors[i % len(colors)], linestyle='--')

        # Add labels to points
        for j, (x, y, dp, ad, mc) in enumerate(zip(ndcg_values, qps_values, doc_pools, ann_docs_values, maxsim_candidates_values)):
            label = f"{dp}, {ad}, {mc}: {x:.2f}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), 
                         ha='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', alpha=1))

    plt.xlabel('NDCG', fontsize=14)
    plt.ylabel('QPS', fontsize=14)
    plt.title('QPS vs NDCG for Pareto-Optimal Points', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    plt.savefig('qps_vs_ndcg_pareto.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
