#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Report variance diagnostics for captured CAA/EM cache rows.")
    parser.add_argument("cache_dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = diagnose(Path(args.cache_dir))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def diagnose(cache_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("current_em_state"):
            rows.append(row)
    current_dims = sorted({key for row in rows for key in (row.get("current_em_state") or {})})
    input_keys = sorted({key for row in rows for key in (row.get("em_v2_inputs") or {})})
    return {
        "schema_version": "activation_rag.caa_em_cache_diagnostic.v1",
        "cache_dir": str(cache_dir),
        "row_count": len(rows),
        "prompt_sections": _counts(row.get("prompt_section_label") for row in rows),
        "current_em_state": _stats(rows, "current_em_state", current_dims),
        "delta_vs_neutral": _stats(rows, "delta_vs_neutral", current_dims),
        "em_v2_inputs": _stats(rows, "em_v2_inputs", input_keys),
    }


def _stats(rows: list[dict[str, Any]], field: str, keys: list[str]) -> dict[str, Any]:
    per_key = {}
    nonconstant = 0
    for key in keys:
        values = [float((row.get(field) or {}).get(key, 0.0)) for row in rows]
        if not values:
            continue
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        value_range = max(values) - min(values)
        if abs(value_range) > 1e-9:
            nonconstant += 1
        per_key[key] = {
            "mean": statistics.mean(values),
            "std": std,
            "min": min(values),
            "max": max(values),
            "range": value_range,
            "unique_rounded_4": len({round(value, 4) for value in values}),
        }
    return {
        "key_count": len(keys),
        "nonconstant_key_count": nonconstant,
        "keys": per_key,
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
