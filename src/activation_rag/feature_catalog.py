from __future__ import annotations

import fnmatch
import json
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureCatalogEntry:
    feature_pattern: str
    semantic_label: str
    categories: tuple[str, ...] = ()
    polarity: str | None = None
    causal_confidence: str | None = None
    validation_status: str | None = None
    causal_effect: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def matches(self, feature_name: str) -> bool:
        return fnmatch.fnmatchcase(feature_name, self.feature_pattern)


@dataclass(frozen=True)
class FeatureCatalog:
    entries: tuple[FeatureCatalogEntry, ...]
    source_path: str
    schema_version: str = "activation_rag.activation_feature_catalog.v1"
    counterfactual_group_count: int = 0
    counterfactual_group_size: int = 0
    counterfactual_seed: int = 13

    def groups_for_feature_names(self, feature_names: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
        names = sorted(str(name) for name in feature_names)
        groups: dict[str, set[str]] = {}
        for entry in self.entries:
            matched = [name for name in names if entry.matches(name)]
            if not matched:
                continue
            _add_group(groups, f"semantic_label:{_slug(entry.semantic_label)}", matched)
            for category in entry.categories:
                _add_group(groups, f"category:{_slug(category)}", matched)
            if entry.polarity:
                _add_group(groups, f"polarity:{_slug(entry.polarity)}", matched)
            if entry.causal_confidence:
                _add_group(groups, f"causal_confidence:{_slug(entry.causal_confidence)}", matched)
            if entry.validation_status:
                _add_group(groups, f"validation_status:{_slug(entry.validation_status)}", matched)
        for index, matched in enumerate(_counterfactual_groups(names, self.counterfactual_group_count, self.counterfactual_group_size, self.counterfactual_seed)):
            _add_group(groups, f"counterfactual:shuffle_{index:02d}", matched)
        return {key: sorted(value) for key, value in sorted(groups.items())}

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.source_path,
            "schema_version": self.schema_version,
            "entry_count": len(self.entries),
            "counterfactual_group_count": self.counterfactual_group_count,
            "counterfactual_group_size": self.counterfactual_group_size,
            "counterfactual_seed": self.counterfactual_seed,
        }


def load_feature_catalog(path: str | Path) -> FeatureCatalog:
    source_path = Path(path)
    if source_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload: dict[str, Any] = {"schema_version": "activation_rag.activation_feature_catalog.v1", "features": rows}
    else:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version") or "")
    if schema_version == "vicuna.rmt_span_selector.feature_manifest.v1":
        entries = tuple(_longmem_entries(row) for row in payload.get("features") or [])
        entries = tuple(entry for pair in entries for entry in pair)
    else:
        entries = tuple(_entry_from_catalog_row(row) for row in payload.get("features") or payload.get("entries") or [])
    entries = tuple(entry for entry in entries if entry.feature_pattern and entry.semantic_label)
    return FeatureCatalog(
        entries=entries,
        source_path=str(source_path),
        schema_version=schema_version or "activation_rag.activation_feature_catalog.v1",
        counterfactual_group_count=int(payload.get("counterfactual_group_count") or 0),
        counterfactual_group_size=int(payload.get("counterfactual_group_size") or 0),
        counterfactual_seed=int(payload.get("counterfactual_seed") or 13),
    )


def _entry_from_catalog_row(row: dict[str, Any]) -> FeatureCatalogEntry:
    pattern = row.get("feature_pattern") or row.get("feature_name") or row.get("name")
    if not pattern and row.get("feature_id") is not None:
        pattern = f"sae.feature.{row['feature_id']}"
    label = row.get("semantic_label") or row.get("label") or row.get("short_label") or pattern
    return FeatureCatalogEntry(
        feature_pattern=str(pattern or ""),
        semantic_label=str(label or ""),
        categories=tuple(str(item) for item in _list(row.get("categories") or row.get("utility_categories"))),
        polarity=_optional_str(row.get("polarity")),
        causal_confidence=_optional_str(row.get("causal_confidence") or row.get("current_status")),
        validation_status=_optional_str(row.get("validation_status") or row.get("current_status") or row.get("status")),
        causal_effect=_optional_float(row.get("causal_effect")),
        provenance={"source_schema": row.get("schema_version")},
    )


def _entry_from_longmem_feature(row: dict[str, Any]) -> FeatureCatalogEntry:
    feature_id = str(row.get("feature_id") or "")
    return FeatureCatalogEntry(
        feature_pattern=f"sae.feature.{feature_id}",
        semantic_label=str(row.get("label") or row.get("short_label") or f"feature {feature_id}"),
        categories=tuple(str(item) for item in _list(row.get("categories"))),
        polarity=_optional_str(row.get("polarity")),
        causal_confidence=_optional_str(row.get("validation_status")),
        validation_status=_optional_str(row.get("validation_status")),
        causal_effect=_optional_float(row.get("causal_effect")),
        provenance={"feature_id": feature_id, "source_schema": "vicuna.rmt_span_selector.feature_manifest.v1"},
    )


def _longmem_entries(row: dict[str, Any]) -> tuple[FeatureCatalogEntry, ...]:
    entry = _entry_from_longmem_feature(row)
    feature_id = str(row.get("feature_id") or "")
    if not feature_id:
        return (entry,)
    return (entry, replace(entry, feature_pattern=feature_id))


def _add_group(groups: dict[str, set[str]], key: str, values: list[str]) -> None:
    groups.setdefault(key, set()).update(values)


def _counterfactual_groups(feature_names: list[str], count: int, size: int, seed: int) -> list[list[str]]:
    if count <= 0 or size <= 0 or not feature_names:
        return []
    rng = random.Random(seed)
    out: list[list[str]] = []
    width = min(size, len(feature_names))
    for _ in range(count):
        out.append(sorted(rng.sample(feature_names, width)))
    return out


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value:
        return [value]
    return []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed
