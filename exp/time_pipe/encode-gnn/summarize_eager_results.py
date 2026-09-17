"""Summarize eager encode/GNN timing logs and plot their time ratios."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = ("cora_node", "pubmed_node", "wikics")
BATCH_NUMS = (1, 5, 10)
DISPLAY_NAMES = {
    "cora_node": "Cora",
    "pubmed_node": "PubMed",
    "wikics": "WikiCS",
}
CSV_FIELDS = (
    "dataset",
    "requested_batch_num",
    "measured_batches",
    "measured_graphs",
    "encoded_node_texts_before_dedup",
    "encoded_edge_texts_before_dedup",
    "encode_seconds",
    "gnn_seconds",
    "measured_total_seconds",
    "encode_percent",
    "gnn_percent",
    "encode_seconds_per_batch",
    "gnn_seconds_per_batch",
    "metric_name",
    "metric_value",
    "checkpoint",
)


def parse_report(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = '{\n  "mode": "eager"'
    start = text.rfind(marker)
    if start < 0:
        raise RuntimeError(f"No eager JSON report found in {log_path}")
    report, _ = json.JSONDecoder().raw_decode(text[start:])
    if not report.get("checkpoint_loaded"):
        raise RuntimeError(f"Checkpoint was not loaded for {log_path}")
    return report


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for task in TASKS:
        for batch_num in BATCH_NUMS:
            report = parse_report(results_dir / f"{task}_batch_{batch_num}.log")
            if report["task_names"] != [task]:
                raise RuntimeError(
                    f"Unexpected task in report: {report['task_names']} (expected {task})"
                )
            if report["requested_batch_num"] != batch_num:
                raise RuntimeError(
                    "Unexpected requested_batch_num in "
                    f"{task}_batch_{batch_num}.log: {report['requested_batch_num']}"
                )
            rows.append({"dataset": task, **report})
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# Eager Encode/GNN Timing Results",
        "",
        "| Dataset | Requested batches | Measured batches | Encode (s) | GNN (s) | Encode (%) | GNN (%) | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {requested_batch_num} | {measured_batches} | "
            "{encode_seconds:.6f} | {gnn_seconds:.6f} | {encode_percent:.2f} | "
            "{gnn_percent:.2f} | {metric_value:.4f} |".format(
                dataset=DISPLAY_NAMES[row["dataset"]], **row
            )
        )
    lines.extend(
        [
            "",
            "Timing excludes model loading, data preparation, warmup, and metric computation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ratios(rows: list[dict], output_path: Path) -> None:
    x_positions = list(range(len(rows)))
    encode_percent = [row["encode_percent"] for row in rows]
    gnn_percent = [row["gnn_percent"] for row in rows]
    labels = [
        f"{DISPLAY_NAMES[row['dataset']]}\n{row['requested_batch_num']} batches"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(
        x_positions,
        encode_percent,
        width=0.72,
        color="#168C86",
        label="LLM encode",
    )
    ax.bar(
        x_positions,
        gnn_percent,
        width=0.72,
        bottom=encode_percent,
        color="#E76F51",
        label="GNN forward",
    )

    for index, row in enumerate(rows):
        ax.text(
            index,
            101.2,
            f"{row['measured_total_seconds']:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            index,
            max(1.5, row["encode_percent"] / 2),
            f"{row['encode_percent']:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    ax.axvline(2.5, color="#A8ADB4", linewidth=0.8)
    ax.axvline(5.5, color="#A8ADB4", linewidth=0.8)
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Share of measured encode + GNN time (%)")
    ax.set_title("Eager Inference Time Ratio by Dataset and Batch Count")
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    rows = collect_rows(results_dir)
    write_csv(rows, results_dir / "summary.csv")
    write_markdown(rows, results_dir / "summary.md")
    plot_ratios(rows, results_dir / "eager_time_ratio.png")
    print(f"Wrote timing summary and plot to {results_dir}")


if __name__ == "__main__":
    main()
