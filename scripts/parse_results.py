#!/usr/bin/env python3
# parse_results.py — export final metrics as per-run CSV rows + per-group joint CSV
import argparse
import fnmatch
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


# --------- helpers ---------


def list_groups() -> List[str]:
    if not LOGS.exists():
        return []
    return sorted([p.name for p in LOGS.iterdir() if p.is_dir()])


def list_runs_in_group(group: str) -> List[str]:
    base = LOGS / group / "runs"
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def ensure_metrics_dir(group: str) -> Path:
    out_dir = LOGS / "metrics" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def newest_run_dir(wandb_dir: Path) -> Optional[Path]:
    """Return the most recent run directory under .../wandb that contains history/summary files."""
    if not wandb_dir.exists():
        return None

    run_dirs = []
    for p in wandb_dir.iterdir():
        if p.is_dir() and p.name.startswith("run-"):
            run_dirs.append(p)

    if not run_dirs:
        return None

    # Sort by name and take the last one (most recent)
    run_dirs.sort(key=lambda x: x.name)
    return run_dirs[-1]


def find_history_and_summary(run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    files_dir = run_dir / "files"
    hist = None
    summ = None
    if (files_dir / "wandb-history.jsonl").exists():
        hist = files_dir / "wandb-history.jsonl"
    elif (run_dir / "wandb-history.jsonl").exists():
        hist = run_dir / "wandb-history.jsonl"
    if (files_dir / "wandb-summary.json").exists():
        summ = files_dir / "wandb-summary.json"
    elif (run_dir / "wandb-summary.json").exists():
        summ = run_dir / "wandb-summary.json"
    return hist, summ


def collect_metric_names_from_history(
    history_path: Path, patterns: Iterable[str]
) -> List[str]:
    wanted = set()
    patts = list(patterns)
    with history_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for key in row.keys():
                for patt in patts:
                    if fnmatch.fnmatch(key, patt):
                        wanted.add(key)
            if i > 1000 and wanted:
                break
    return sorted(wanted)


def last_non_null(
    history_path: Path, metric_keys: Iterable[str]
) -> Dict[str, Optional[float]]:
    keys = set(metric_keys)
    last: Dict[str, Optional[float]] = {k: None for k in keys}
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for k in keys:
                if k in row and row[k] is not None:
                    last[k] = row[k]
    return last


def fill_from_summary(
    summary_path: Path, metric_keys: Iterable[str], known: Dict[str, Optional[float]]
) -> Dict[str, Optional[float]]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    for k in metric_keys:
        if known.get(k) is None and k in data and data[k] is not None:
            known[k] = data[k]
    return known


def expand_patterns_to_keys(
    history_path: Optional[Path], summary_path: Optional[Path], patterns: List[str]
) -> List[str]:
    keys = set()
    if history_path and history_path.exists():
        keys.update(collect_metric_names_from_history(history_path, patterns))
    elif summary_path and summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for key in data.keys():
            for patt in patterns:
                if fnmatch.fnmatch(key, patt):
                    keys.add(key)
    return sorted(keys) if keys else patterns


def format_table(rows: List[Tuple[str, Optional[float]]]) -> str:
    name_w = max(6, *(len(r[0]) for r in rows)) if rows else 10
    header = f"{'metric'.ljust(name_w)}  {'final_value':>12}"
    bar = "-" * (name_w + 2 + 12)
    lines = [header, bar]
    for m, v in rows:
        v_str = f"{v:.3g}" if isinstance(v, (int, float)) else "—"
        lines.append(f"{m.ljust(name_w)}  {v_str:>12}")
    return "\n".join(lines)


# --------- core ---------


def pick_run_name_for_group(
    group: str, user_run_token: str, available_runs: List[str]
) -> Optional[str]:
    """If user supplies 'wan' and group is 'dl3dv', return 'dl3dv_wan' when available."""
    prefixed = f"{group}_{user_run_token}"
    if user_run_token in available_runs:
        return user_run_token
    if prefixed in available_runs:
        return prefixed
    return None


def process_run(
    group: str, run_name_with_group: str, keys_for_group: List[str]
) -> Tuple[str, Dict[str, Optional[float]]]:
    base = LOGS / group / "runs" / run_name_with_group / "wandb"
    rd = newest_run_dir(base)
    if not rd:
        return (f"[!] No W&B run directory found under: {base}", {})

    history_path, summary_path = find_history_and_summary(rd)

    results: Dict[str, Optional[float]] = {k: None for k in keys_for_group}
    # Fill values from history then summary
    if history_path and history_path.exists():
        results.update(last_non_null(history_path, keys_for_group))
    if summary_path and summary_path.exists():
        results = fill_from_summary(summary_path, keys_for_group, results)

    rows = [(k, results[k]) for k in keys_for_group]
    header = f"\n=== {group}/{run_name_with_group}  ({rd.name}) ==="
    return (header + "\n" + format_table(rows), results)


def main():
    parser = argparse.ArgumentParser(
        description="Print and export final metric values from local W&B logs (CSV per run + joint)."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="dl3dv,co3d",
        help="Comma-separated group names (default: dl3dv,co3d)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="wan,cogvideox",
        help="Comma-separated run tokens (auto-prefixed with group).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="val/Auc_30,val/pmap_mse_aligned,val/loss_depth",
        help="Comma-separated metric patterns (wildcards ok).",
    )
    args = parser.parse_args()

    all_groups = set(list_groups())
    want_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    missing_groups = [g for g in want_groups if g not in all_groups]
    for g in missing_groups:
        print(
            f"[!] Group '{g}' not found. Available groups under ./logs: {', '.join(sorted(all_groups)) or '(none)'}"
        )
    want_groups = [g for g in want_groups if g in all_groups]
    if not want_groups:
        return

    patterns = [m.strip() for m in args.metrics.split(",") if m.strip()]
    run_tokens = [r.strip() for r in args.runs.split(",") if r.strip()]

    for g in want_groups:
        available_runs = list_runs_in_group(g)
        available_set = set(available_runs)
        chosen_runs = []
        # First pass: determine which runs we'll process
        for token in run_tokens:
            chosen = pick_run_name_for_group(g, token, available_runs)
            if not chosen:
                print(
                    f"[!] Run token '{token}' not found in group '{g}'. Tried '{token}' and '{g}_{token}'. "
                    f"Available runs: {', '.join(sorted(available_set)) or '(none)'}"
                )
                continue
            chosen_runs.append(chosen)

        if not chosen_runs:
            continue

        # Build a consistent header (union of metric keys across chosen runs), ordered by pattern-order then lexicographic
        union_keys = set()
        for chosen in chosen_runs:
            base = LOGS / g / "runs" / chosen / "wandb"
            rd = newest_run_dir(base)
            if not rd:
                continue
            hist, summ = find_history_and_summary(rd)
            keys = expand_patterns_to_keys(hist, summ, patterns)
            union_keys.update(keys)

        if not union_keys:
            print(f"[!] No matching metrics found for group '{g}'.")
            continue

        def patt_order(key: str) -> int:
            for i, patt in enumerate(patterns):
                if fnmatch.fnmatch(key, patt):
                    return i
            return len(patterns) + 1

        keys_for_group = sorted(union_keys, key=lambda k: (patt_order(k), k))

        # Accumulate rows for per-group joint CSV
        joint_rows = []
        joint_header = ["run"] + keys_for_group

        # Second pass: process each run and write CSV with a single row: run + metrics
        for chosen in chosen_runs:
            console_text, metrics_map = process_run(g, chosen, keys_for_group)
            print(console_text)

            # Row dict with first column 'run' and then metrics in keys_for_group order
            run_label = chosen[len(g) + 1 :] if chosen.startswith(f"{g}_") else chosen
            out_dir = ensure_metrics_dir(g)
            out_path = out_dir / f"{chosen}.csv"
            try:
                with out_path.open("w", encoding="utf-8") as f:
                    # header
                    f.write("run," + ",".join(keys_for_group) + "\n")

                    # row (empty for missing)
                    def fmt(v):
                        if v is None:
                            return ""
                        if isinstance(v, (int, float)):
                            return f"{v:.10g}"
                        return str(v)

                    row_vals = [fmt(metrics_map.get(k, None)) for k in keys_for_group]
                    f.write(",".join([run_label] + row_vals) + "\n")
                joint_rows.append([run_label] + row_vals)
            except Exception as e:
                print(f"[!] Failed to write {out_path}: {e}")

        # Emit per-group joint CSV
        try:
            out_dir = ensure_metrics_dir(g)
            joint_path = out_dir / "joint.csv"
            with joint_path.open("w", encoding="utf-8") as f:
                f.write(",".join(joint_header) + "\n")
                for row in joint_rows:
                    f.write(",".join(row) + "\n")
            print(f"[i] Wrote joint CSV: {joint_path}")
        except Exception as e:
            print(f"[!] Failed to write joint CSV for group '{g}': {e}")


if __name__ == "__main__":
    main()
