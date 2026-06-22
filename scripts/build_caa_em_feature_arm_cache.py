#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


ARMS = ("caa8_only", "caa8_aggregates", "sae_plus_caa8", "caa8_matched_control")
DIMENSIONS = (
    "constraint_imposed",
    "final_answer_readiness",
    "hair",
    "plan_formation",
    "repair_readiness",
    "runtime_failure_pressure",
    "stall_looping",
    "state_carryover",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAA/EM feature-arm telemetry caches for existing direct-blend trainer.")
    parser.add_argument("--caa-cache-dir", required=True)
    parser.add_argument("--out-cache-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--sae-cache-dir")
    parser.add_argument("--sae-feature-manifest")
    parser.add_argument("--seed", default="13")
    parser.add_argument("--allow-constant-caa", action="store_true")
    args = parser.parse_args()

    summary = build_arm_cache(
        caa_cache_dir=Path(args.caa_cache_dir),
        out_cache_dir=Path(args.out_cache_dir),
        manifest_out=Path(args.manifest_out),
        arm=str(args.arm),
        sae_cache_dir=Path(args.sae_cache_dir) if args.sae_cache_dir else None,
        sae_feature_manifest=Path(args.sae_feature_manifest) if args.sae_feature_manifest else None,
        seed=str(args.seed),
        allow_constant_caa=bool(args.allow_constant_caa),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arm_cache(
    *,
    caa_cache_dir: Path,
    out_cache_dir: Path,
    manifest_out: Path,
    arm: str,
    sae_cache_dir: Path | None,
    sae_feature_manifest: Path | None,
    seed: str,
    allow_constant_caa: bool,
) -> dict[str, Any]:
    caa_rows = _load_rows(caa_cache_dir, require_caa=True)
    variance = _caa_variance_summary(caa_rows)
    if not allow_constant_caa and variance["nonconstant_current_dimensions"] == 0:
        raise ValueError(
            "CAA/EM cache has zero variance across current_em_state dimensions; "
            "refusing to build a misleading feature arm"
        )
    sae_rows = _load_rows(sae_cache_dir, require_caa=False) if sae_cache_dir else {}
    if arm == "sae_plus_caa8" and not sae_rows:
        raise ValueError("--sae-cache-dir is required for sae_plus_caa8")
    control_map = _control_map(caa_rows, seed=seed) if arm == "caa8_matched_control" else {}
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    feature_ids: list[str] = []
    written = 0
    skipped = 0
    for chunk_id, caa_row in sorted(caa_rows.items()):
        source_caa = control_map.get(chunk_id, caa_row)
        features = _caa_features(source_caa, aggregate_only=arm == "caa8_aggregates")
        row = dict(caa_row)
        if arm == "sae_plus_caa8":
            sae_row = sae_rows.get(chunk_id)
            if sae_row is None:
                skipped += 1
                continue
            row = dict(sae_row)
            row.update(
                {
                    "current_em_state": caa_row.get("current_em_state") or {},
                    "neutral_baseline_state": caa_row.get("neutral_baseline_state") or {},
                    "prior_current_state": caa_row.get("prior_current_state") or {},
                    "delta_vs_neutral": caa_row.get("delta_vs_neutral") or {},
                    "delta_vs_current": caa_row.get("delta_vs_current") or {},
                    "saturation": caa_row.get("saturation") or {},
                    "residual_headroom": caa_row.get("residual_headroom") or {},
                    "positive_mass": caa_row.get("positive_mass", 0.0),
                    "negative_mass": caa_row.get("negative_mass", 0.0),
                    "total_mass": caa_row.get("total_mass", 0.0),
                    "signed_balance": caa_row.get("signed_balance", 0.0),
                }
            )
            combined = {str(key): float(value) for key, value in (sae_row.get("sae_feature_values") or {}).items()}
            combined.update(features)
            features = combined
        row["provider_id"] = f"{row.get('provider_id', 'qwen-caa8')}.{arm}"
        row["normalization_policy"] = f"{row.get('normalization_policy', 'qwen3_8head_neutral_baseline_delta_v1')}::{arm}"
        row["sae_feature_values"] = features
        row["sae_delta_vs_neutral"] = dict(features)
        row["sae_delta_vs_current"] = dict(features)
        row["sae_feature_mask"] = {key: value != 0.0 for key, value in features.items()}
        row["telemetry_valid"] = bool(features)
        row["invalid_reason"] = None if features else "no_arm_features"
        row.setdefault("provenance", {})
        row["provenance"] = dict(row["provenance"])
        row["provenance"]["caa_feature_arm"] = arm
        row["provenance"]["source_caa_chunk_id"] = str(source_caa.get("chunk_id")) if source_caa is not caa_row else chunk_id
        (out_cache_dir / f"{chunk_id}.json").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        written += 1
        if not feature_ids:
            feature_ids = list(features.keys())
        else:
            for key in features:
                if key not in feature_ids:
                    feature_ids.append(key)
    if arm == "sae_plus_caa8" and sae_feature_manifest and sae_feature_manifest.exists():
        feature_ids = _ordered_sae_ids(sae_feature_manifest) + [key for key in feature_ids if key.startswith("caa.")]
    manifest = _manifest(arm, feature_ids)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "activation_rag.caa8_feature_arm_cache_summary.v1",
        "arm": arm,
        "caa_cache_dir": str(caa_cache_dir),
        "sae_cache_dir": str(sae_cache_dir) if sae_cache_dir else None,
        "out_cache_dir": str(out_cache_dir),
        "manifest_out": str(manifest_out),
        "input_caa_rows": len(caa_rows),
        "input_sae_rows": len(sae_rows),
        "written_rows": written,
        "skipped_rows": skipped,
        "feature_count": len(feature_ids),
        "caa_variance_summary": variance,
    }
    (out_cache_dir / "arm-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_rows(cache_dir: Path | None, *, require_caa: bool) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if cache_dir is None:
        return rows
    for path in sorted(cache_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("telemetry_valid", True) is False or row.get("invalid_reason"):
            continue
        if require_caa and not row.get("current_em_state"):
            continue
        if not require_caa and not row.get("sae_feature_values"):
            continue
        rows[str(row["chunk_id"])] = row
    return rows


def _caa_features(row: dict[str, Any], *, aggregate_only: bool) -> dict[str, float]:
    current = _float_map(row.get("current_em_state"))
    delta_neutral = _float_map(row.get("delta_vs_neutral"))
    delta_current = _float_map(row.get("delta_vs_current"))
    saturation = _float_map(row.get("saturation"))
    headroom = _float_map(row.get("residual_headroom"))
    features: dict[str, float] = {}
    if not aggregate_only:
        for dim in DIMENSIONS:
            features[f"caa.current.{dim}"] = current.get(dim, 0.0)
            features[f"caa.delta_neutral.{dim}"] = delta_neutral.get(dim, 0.0)
            features[f"caa.delta_current.{dim}"] = delta_current.get(dim, 0.0)
            features[f"caa.saturation.{dim}"] = saturation.get(dim, 0.0)
            features[f"caa.headroom.{dim}"] = headroom.get(dim, 0.0)
    dn = [delta_neutral.get(dim, 0.0) for dim in DIMENSIONS]
    dc = [delta_current.get(dim, 0.0) for dim in DIMENSIONS]
    cur = [current.get(dim, 0.0) for dim in DIMENSIONS]
    features.update(
        {
            "caa.aggregate.current_l2": _l2(cur),
            "caa.aggregate.delta_neutral_l2": _l2(dn),
            "caa.aggregate.delta_current_l2": _l2(dc),
            "caa.aggregate.current_mean": sum(cur) / max(1, len(cur)),
            "caa.aggregate.delta_neutral_mean": sum(dn) / max(1, len(dn)),
            "caa.aggregate.delta_neutral_abs_sum": sum(abs(value) for value in dn),
            "caa.aggregate.positive_mass": float(row.get("positive_mass") or 0.0),
            "caa.aggregate.negative_mass": float(row.get("negative_mass") or 0.0),
            "caa.aggregate.total_mass": float(row.get("total_mass") or 0.0),
            "caa.aggregate.signed_balance": float(row.get("signed_balance") or 0.0),
        }
    )
    return features


def _control_map(rows: dict[str, dict[str, Any]], *, seed: str) -> dict[str, dict[str, Any]]:
    keys = sorted(rows)
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == keys:
        shuffled = shuffled[1:] + shuffled[:1]
    return {key: rows[shuffled[index]] for index, key in enumerate(keys)}


def _caa_variance_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_dim: dict[str, dict[str, float]] = {}
    for dim in DIMENSIONS:
        values = [float((row.get("current_em_state") or {}).get(dim, 0.0)) for row in rows.values()]
        if not values:
            per_dim[dim] = {"min": 0.0, "max": 0.0, "range": 0.0}
            continue
        min_value = min(values)
        max_value = max(values)
        per_dim[dim] = {"min": min_value, "max": max_value, "range": max_value - min_value}
    return {
        "row_count": len(rows),
        "dimensions": per_dim,
        "nonconstant_current_dimensions": sum(1 for item in per_dim.values() if abs(float(item["range"])) > 1e-9),
    }


def _ordered_sae_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("features") or []
    ids = [str(row["feature_id"]) for row in rows if row.get("feature_id") is not None]
    if ids:
        return ids
    return [str(value) for value in payload.get("sae_feature_ids") or []]


def _manifest(arm: str, feature_ids: list[str]) -> dict[str, Any]:
    features = []
    for feature_id in feature_ids:
        if feature_id.startswith("caa.aggregate."):
            categories = ["caa", "aggregate"]
        elif feature_id.startswith("caa."):
            categories = ["caa", "em_v2_8head"]
        else:
            categories = ["sae", "core245"]
        features.append(
            {
                "feature_id": feature_id,
                "label": feature_id,
                "categories": categories,
                "causal_effect": 0.0,
            }
        )
    return {
        "schema_version": "activation_rag.caa8_feature_manifest.v1",
        "feature_set_id": f"{arm}_qwen3_8head_caa_em_v2",
        "features": features,
    }


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if _finite(raw)}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


if __name__ == "__main__":
    main()
