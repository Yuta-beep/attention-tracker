import argparse
import json
from pathlib import Path

import numpy as np

from .plots import plot_dataset_delta_heatmaps, plot_similarity_heatmaps
from .runs import create_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-result", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-title", default="head-generalization")
    return parser.parse_args()


def _parse_dataset_arg(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError("Each --dataset-result must be name=path.json")
    name, raw_path = item.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_delta_matrix(result_payload: dict) -> np.ndarray:
    deltas = []
    for case in result_payload["cases"]:
        normal = np.asarray(case["normal_instruction_scores"], dtype=float)
        attack = np.asarray(case["attack_instruction_scores"], dtype=float)
        deltas.append(normal - attack)
    return np.mean(np.stack(deltas, axis=0), axis=0)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _topk_overlap(x: np.ndarray, y: np.ndarray, top_k: int) -> dict:
    k = min(top_k, len(x))
    top_x = set(np.argsort(x)[-k:].tolist())
    top_y = set(np.argsort(y)[-k:].tolist())
    intersection = len(top_x & top_y)
    union = len(top_x | top_y)
    return {
        "top_k": k,
        "intersection": intersection,
        "overlap_ratio": intersection / k if k else 0.0,
        "jaccard": intersection / union if union else 0.0,
    }


def _pairwise_stats(dataset_vectors: dict[str, np.ndarray], top_k: int) -> tuple[list[dict], dict[str, list[list[float]]]]:
    names = list(dataset_vectors)
    pearson_matrix = []
    spearman_matrix = []
    overlap_matrix = []
    jaccard_matrix = []
    pairs = []

    for name_a in names:
        pearson_row = []
        spearman_row = []
        overlap_row = []
        jaccard_row = []
        for name_b in names:
            x = dataset_vectors[name_a]
            y = dataset_vectors[name_b]
            pearson_value = _pearson(x, y)
            spearman_value = _spearman(x, y)
            overlap = _topk_overlap(x, y, top_k)
            pearson_row.append(pearson_value)
            spearman_row.append(spearman_value)
            overlap_row.append(overlap["overlap_ratio"])
            jaccard_row.append(overlap["jaccard"])
            if name_a < name_b:
                pairs.append(
                    {
                        "dataset_a": name_a,
                        "dataset_b": name_b,
                        "pearson": pearson_value,
                        "spearman": spearman_value,
                        **overlap,
                    }
                )
        pearson_matrix.append(pearson_row)
        spearman_matrix.append(spearman_row)
        overlap_matrix.append(overlap_row)
        jaccard_matrix.append(jaccard_row)

    matrices = {
        "names": names,
        "pearson": pearson_matrix,
        "spearman": spearman_matrix,
        "topk_overlap_ratio": overlap_matrix,
        "jaccard": jaccard_matrix,
    }
    return pairs, matrices


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.output_root, args.experiment_title)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    dataset_vectors = {}
    dataset_matrices = {}

    for item in args.dataset_result:
        name, path = _parse_dataset_arg(item)
        payload = _load_result(path)
        delta_matrix = _mean_delta_matrix(payload)
        datasets[name] = {
            "path": str(path),
            "num_cases": len(payload["cases"]),
            "model_key": payload["model_key"],
            "model_id": payload["model_id"],
        }
        dataset_matrices[name] = delta_matrix
        dataset_vectors[name] = delta_matrix.reshape(-1)

    pairwise, matrices = _pairwise_stats(dataset_vectors, args.top_k)
    top_heads = {}
    for name, vector in dataset_vectors.items():
        top_indices = np.argsort(vector)[-min(args.top_k, len(vector)) :][::-1]
        num_heads = dataset_matrices[name].shape[1]
        top_heads[name] = [
            {
                "rank": rank + 1,
                "layer": int(index // num_heads),
                "head": int(index % num_heads),
                "delta": float(vector[index]),
            }
            for rank, index in enumerate(top_indices)
        ]

    plot_dataset_delta_heatmaps(dataset_matrices, plots_dir / "dataset_delta_heatmaps.png")
    plot_similarity_heatmaps(matrices, plots_dir / "dataset_similarity.png")

    payload = {
        "experiment_title": args.experiment_title,
        "run_dir": str(run_dir),
        "top_k": args.top_k,
        "datasets": datasets,
        "pairwise": pairwise,
        "matrices": matrices,
        "top_heads": top_heads,
    }
    (run_dir / "head_generalization.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "Generated files:\n"
        "- head_generalization.json\n"
        "- plots/dataset_delta_heatmaps.png\n"
        "- plots/dataset_similarity.png\n",
        encoding="utf-8",
    )
    print(f"Saved summary: {(run_dir / 'head_generalization.json').resolve()}")
    print(f"Saved plots: {plots_dir.resolve()}")
    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
