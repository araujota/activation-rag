from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

from activation_rag.schema import ActivationRecord, ChunkRecord, stable_hash


EM_DIMENSIONS = (
    "clarification_pressure",
    "tool_compulsion",
    "verification_pressure",
    "repair_readiness",
    "plan_formation",
    "termination_readiness",
)


class TelemetryProvider(Protocol):
    provider_id: str
    model_id: str

    def capture_prefill(self, chunks: list[ChunkRecord]) -> list[ActivationRecord]:
        ...


class MockTelemetryProvider:
    def __init__(
        self,
        neutral_baseline_state: dict[str, float] | None = None,
        prior_current_state: dict[str, float] | None = None,
        provider_id: str = "mock-selector-telemetry",
        model_id: str = "mock-prefill-model",
        site_id: str = "mock_resid_prefill",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.site_id = site_id
        self.neutral_baseline_state = neutral_baseline_state or {key: 0.0 for key in EM_DIMENSIONS}
        self.prior_current_state = prior_current_state or {key: 0.0 for key in EM_DIMENSIONS}

    def capture_prefill(self, chunks: list[ChunkRecord]) -> list[ActivationRecord]:
        capture_run_id = stable_hash("|".join(chunk.chunk_id for chunk in chunks) or "empty", 24)
        return [self._capture_chunk(chunk, capture_run_id) for chunk in chunks]

    def _capture_chunk(self, chunk: ChunkRecord, capture_run_id: str) -> ActivationRecord:
        current = self._score_text(chunk.text)
        all_keys = sorted(set(current) | set(self.neutral_baseline_state) | set(self.prior_current_state))
        neutral = {key: float(self.neutral_baseline_state.get(key, 0.0)) for key in all_keys}
        prior = {key: float(self.prior_current_state.get(key, 0.0)) for key in all_keys}
        current_full = {key: float(current.get(key, 0.0)) for key in all_keys}
        delta_neutral = {key: current_full[key] - neutral[key] for key in all_keys}
        delta_current = {key: current_full[key] - prior[key] for key in all_keys}
        saturation = {key: min(1.0, max(0.0, abs(current_full[key]))) for key in all_keys}
        residual_headroom = {key: max(0.0, 1.0 - saturation[key]) for key in all_keys}
        positive_mass = sum(value for value in delta_neutral.values() if value > 0.0)
        negative_mass = sum(-value for value in delta_neutral.values() if value < 0.0)
        total_mass = positive_mass + negative_mass
        signed_balance = 0.0 if total_mass == 0.0 else (positive_mass - negative_mass) / total_mass
        sae_values = self._mock_sae_features(chunk.text, current_full)
        sae_delta_neutral = dict(sae_values)
        sae_delta_current = {key: value * 0.5 for key, value in sae_values.items()}
        sae_mask = {key: value != 0.0 for key, value in sae_values.items()}

        return ActivationRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            capture_run_id=capture_run_id,
            provider_id=self.provider_id,
            model_id=self.model_id,
            site_id=self.site_id,
            hook_name="mock.hook_resid_pre",
            layer_index=0,
            layer_selection_policy="mock_single_site_l0_resid_prefill",
            prompt_template_id="raw_chunk_v1",
            prompt_template_hash=stable_hash("raw_chunk_v1:${chunk_text}", 24),
            normalization_policy="mock_unit_interval_keyword_scores",
            token_start=0,
            token_end=chunk.token_count_estimate,
            current_em_state=current_full,
            neutral_baseline_state=neutral,
            prior_current_state=prior,
            delta_vs_neutral=delta_neutral,
            delta_vs_current=delta_current,
            saturation=saturation,
            residual_headroom=residual_headroom,
            positive_mass=positive_mass,
            negative_mass=negative_mass,
            total_mass=total_mass,
            signed_balance=signed_balance,
            sae_feature_values=sae_values,
            sae_delta_vs_neutral=sae_delta_neutral,
            sae_delta_vs_current=sae_delta_current,
            sae_feature_mask=sae_mask,
            sae_novelty=sum(1.0 for active in sae_mask.values() if active) / max(1, len(sae_mask)),
            sae_overlap_with_memory=None,
            provenance={
                "chunk_text_hash": chunk.text_hash,
                "chunker": chunk.chunker,
                "provider_mode": "deterministic_keyword_mock",
            },
        )

    def _score_text(self, text: str) -> dict[str, float]:
        tokens = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
        return {
            "clarification_pressure": self._has(tokens, "clarify", "uncertain", "question"),
            "tool_compulsion": self._has(tokens, "tool", "search", "lookup", "external"),
            "verification_pressure": self._has(tokens, "verify", "verification", "evidence", "checked", "carefully"),
            "repair_readiness": self._has(tokens, "repair", "fix", "bug", "failing"),
            "plan_formation": self._has(tokens, "plan", "steps", "design", "architecture"),
            "termination_readiness": self._has(tokens, "termination", "final", "answer", "concise"),
        }

    def _mock_sae_features(self, text: str, em_state: dict[str, float]) -> dict[str, float]:
        lowered = text.lower()
        return {
            "mock:verification": em_state.get("verification_pressure", 0.0),
            "mock:planning": em_state.get("plan_formation", 0.0),
            "mock:termination": em_state.get("termination_readiness", 0.0),
            "mock:length": min(1.0, len(lowered.split()) / 64.0),
        }

    def _has(self, tokens: set[str], *needles: str) -> float:
        hits = sum(1 for needle in needles if needle in tokens)
        return min(1.0, hits / max(1, len(needles)))


class CommandPrefillTelemetryProvider:
    """Adapter for real prefill telemetry emitted by an external capture command."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        provider_id: str,
        model_id: str,
        site_id: str,
        layer_selection_policy: str,
        prompt_template_id: str,
        normalization_policy: str,
        prompt_template: str = "{chunk_text}",
        timeout_seconds: int = 3600,
        work_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = tuple(command)
        self.provider_id = provider_id
        self.model_id = model_id
        self.site_id = site_id
        self.layer_selection_policy = layer_selection_policy
        self.prompt_template_id = prompt_template_id
        self.prompt_template = prompt_template
        self.prompt_template_hash = stable_hash(prompt_template, 24)
        self.normalization_policy = normalization_policy
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir) if work_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def capture_prefill(self, chunks: list[ChunkRecord]) -> list[ActivationRecord]:
        capture_run_id = stable_hash(
            "|".join([self.provider_id, self.model_id, self.prompt_template_hash, *(chunk.chunk_id for chunk in chunks)]) or "empty",
            24,
        )
        cached_rows: dict[str, dict[str, Any]] = {}
        missing_chunks: list[ChunkRecord] = []
        for chunk in chunks:
            cached = self._read_cache_row(chunk)
            if cached is None:
                missing_chunks.append(chunk)
            else:
                cached_rows[chunk.chunk_id] = cached

        captured_rows: list[dict[str, Any]] = []
        if missing_chunks:
            with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp:
                root = Path(tmp)
                input_path = root / "prefill-input.jsonl"
                output_path = root / "prefill-output.jsonl"
                manifest_path = root / "prefill-manifest.json"
                self._write_input(input_path, manifest_path, missing_chunks, capture_run_id)
                self._run_command(input_path=input_path, output_path=output_path, manifest_path=manifest_path)
                captured_rows = _read_jsonl(output_path)
            for row in captured_rows:
                chunk_id = str(row.get("chunk_id") or row.get("span_id"))
                chunk = next((candidate for candidate in missing_chunks if candidate.chunk_id == chunk_id), None)
                if chunk is not None:
                    self._write_cache_row(chunk, row)
        rows = [*cached_rows.values(), *captured_rows]
        records_by_chunk = {
            str(row.get("chunk_id") or row.get("span_id")): self._row_to_record(row, chunks, capture_run_id)
            for row in rows
        }
        missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in records_by_chunk]
        if missing:
            raise ValueError(f"capture command did not return telemetry for chunks: {missing[:5]}")
        return [records_by_chunk[chunk.chunk_id] for chunk in chunks]

    def _cache_key(self, chunk: ChunkRecord) -> str:
        return stable_hash(
            "|".join(
                [
                    self.provider_id,
                    self.model_id,
                    self.site_id,
                    self.layer_selection_policy,
                    self.prompt_template_hash,
                    self.normalization_policy,
                    chunk.chunk_id,
                    chunk.text_hash,
                ]
            ),
            48,
        )

    def _cache_path(self, chunk: ChunkRecord) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._cache_key(chunk)}.json"

    def _read_cache_row(self, chunk: ChunkRecord) -> dict[str, Any] | None:
        path = self._cache_path(chunk)
        if path is None or not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if not _is_valid_prefill_row(row):
            return None
        provenance = dict(row.get("provenance") or {})
        provenance["cache_hit"] = True
        row["provenance"] = provenance
        return row

    def _write_cache_row(self, chunk: ChunkRecord, row: dict[str, Any]) -> None:
        path = self._cache_path(chunk)
        if path is None:
            return
        if not _is_valid_prefill_row(row):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(row, sort_keys=True))
        provenance = dict(payload.get("provenance") or {})
        provenance["cache_key"] = self._cache_key(chunk)
        payload["provenance"] = provenance
        path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")

    def _write_input(self, input_path: Path, manifest_path: Path, chunks: list[ChunkRecord], capture_run_id: str) -> None:
        manifest = {
            "schema_version": "activation_rag.prefill_capture_manifest.v1",
            "capture_run_id": capture_run_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "site_id": self.site_id,
            "layer_selection_policy": self.layer_selection_policy,
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_hash": self.prompt_template_hash,
            "normalization_policy": self.normalization_policy,
            "capture_phase": "prefill",
            "generation_disabled": True,
            "chunk_count": len(chunks),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with input_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                row = {
                    "schema_version": "activation_rag.prefill_capture_request.v1",
                    "capture_run_id": capture_run_id,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "text_hash": chunk.text_hash,
                    "token_count_estimate": chunk.token_count_estimate,
                    "prompt_text": self.prompt_template.replace("{chunk_text}", chunk.text),
                    "provider_id": self.provider_id,
                    "model_id": self.model_id,
                    "site_id": self.site_id,
                    "layer_selection_policy": self.layer_selection_policy,
                    "prompt_template_id": self.prompt_template_id,
                    "prompt_template_hash": self.prompt_template_hash,
                    "requested_prompt_section_label": "query" if chunk.document_id == "query" else "document_chunk",
                    "normalization_policy": self.normalization_policy,
                    "capture_phase": "prefill",
                    "generation_disabled": True,
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def _run_command(self, *, input_path: Path, output_path: Path, manifest_path: Path) -> None:
        replacements = {
            "input_jsonl": str(input_path),
            "output_jsonl": str(output_path),
            "manifest_json": str(manifest_path),
        }
        command = [part.format(**replacements) for part in self.command]
        completed = subprocess.run(
            command,
            cwd=str(self.work_dir) if self.work_dir else None,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "prefill telemetry command failed "
                f"with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not output_path.exists():
            raise FileNotFoundError(f"prefill telemetry command did not create {output_path}")

    def _row_to_record(self, row: dict[str, Any], chunks: list[ChunkRecord], capture_run_id: str) -> ActivationRecord:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        chunk_id = str(row.get("chunk_id") or row.get("span_id") or "")
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"capture command returned unknown chunk_id: {chunk_id!r}")
        if not _is_valid_prefill_row(row):
            reason = row.get("invalid_reason") or "missing prefill activation features"
            raise ValueError(f"telemetry row for {chunk_id} is invalid: {reason}")
        provenance = dict(row.get("provenance") or {})
        capture_phase = row.get("capture_phase") or provenance.get("capture_phase") or "prefill"
        generation_disabled = row.get("generation_disabled")
        if generation_disabled is None:
            generation_disabled = provenance.get("generation_disabled", True)
        prefill_only_extracted = bool(row.get("prefill_only_extracted") or provenance.get("prefill_only_extracted", False))
        if capture_phase != "prefill":
            raise ValueError(f"telemetry row for {chunk_id} was not prefill-only: {capture_phase!r}")
        if generation_disabled is False and not prefill_only_extracted:
            raise ValueError(f"telemetry row for {chunk_id} reports generation was enabled")
        provenance.setdefault("capture_phase", "prefill")
        provenance.setdefault("generation_disabled", bool(generation_disabled))
        if row.get("requested_prompt_section_label") is not None:
            provenance.setdefault("requested_prompt_section_label", str(row.get("requested_prompt_section_label")))
        if row.get("prompt_section_label") is not None:
            provenance.setdefault("prompt_section_label", str(row.get("prompt_section_label")))
        if row.get("raw_prompt_section_label") is not None:
            provenance.setdefault("raw_prompt_section_label", str(row.get("raw_prompt_section_label")))
        if prefill_only_extracted:
            provenance.setdefault("prefill_only_extracted", True)
        provenance.setdefault("chunk_text_hash", chunk.text_hash)
        provenance.setdefault("chunker", chunk.chunker)

        current = _float_map(row.get("current_em_state") or row.get("current_em") or row.get("absolute_em"))
        neutral = _float_map(row.get("neutral_baseline_state") or row.get("neutral_baseline_em") or row.get("baseline_em"))
        prior = _float_map(row.get("prior_current_state") or row.get("prior_current_em") or row.get("current_em_baseline"))
        delta_neutral = _float_map(row.get("delta_vs_neutral") or row.get("em_delta_vs_neutral") or row.get("span_delta_vs_neutral") or row.get("delta"))
        delta_current = _float_map(row.get("delta_vs_current") or row.get("em_delta_vs_current") or row.get("span_delta_vs_current"))
        if not delta_neutral and current and neutral:
            delta_neutral = {key: current.get(key, 0.0) - neutral.get(key, 0.0) for key in sorted(set(current) | set(neutral))}
        if not delta_current and current and prior:
            delta_current = {key: current.get(key, 0.0) - prior.get(key, 0.0) for key in sorted(set(current) | set(prior))}

        saturation = _float_map(row.get("saturation"))
        if not saturation:
            saturation = {key: min(1.0, abs(value)) for key, value in current.items()}
        residual_headroom = _float_map(row.get("residual_headroom") or row.get("sae_residual_headroom"))
        if not residual_headroom:
            residual_headroom = {key: max(0.0, 1.0 - saturation.get(key, 0.0)) for key in sorted(saturation)}

        positive_mass = _optional_float(row.get("positive_mass"))
        negative_mass = _optional_float(row.get("negative_mass"))
        if positive_mass is None:
            positive_mass = sum(value for value in delta_neutral.values() if value > 0.0)
        if negative_mass is None:
            negative_mass = sum(-value for value in delta_neutral.values() if value < 0.0)
        total_mass = _optional_float(row.get("total_mass"))
        if total_mass is None:
            total_mass = positive_mass + negative_mass
        signed_balance = _optional_float(row.get("signed_balance"))
        if signed_balance is None:
            signed_balance = 0.0 if total_mass == 0.0 else (positive_mass - negative_mass) / total_mass

        sae_values = _float_map(row.get("sae_feature_values"))
        sae_delta_neutral = _float_map(row.get("sae_delta_vs_neutral"))
        sae_delta_current = _float_map(row.get("sae_delta_vs_current"))
        sae_mask = _bool_map(row.get("sae_feature_mask"))
        if not sae_mask:
            sae_mask = {key: value != 0.0 for key, value in sae_values.items()}

        return ActivationRecord(
            chunk_id=chunk.chunk_id,
            document_id=str(row.get("document_id") or chunk.document_id),
            capture_run_id=str(row.get("capture_run_id") or capture_run_id),
            provider_id=str(row.get("provider_id") or self.provider_id),
            model_id=str(row.get("model_id") or self.model_id),
            site_id=str(row.get("site_id") or self.site_id),
            hook_name=row.get("hook_name"),
            layer_index=_optional_int(row.get("layer_index")),
            layer_selection_policy=str(row.get("layer_selection_policy") or self.layer_selection_policy),
            prompt_template_id=str(row.get("prompt_template_id") or self.prompt_template_id),
            prompt_template_hash=str(row.get("prompt_template_hash") or self.prompt_template_hash),
            normalization_policy=str(row.get("normalization_policy") or self.normalization_policy),
            token_start=_optional_int(row.get("token_start")) or 0,
            token_end=_optional_int(row.get("token_end")) or chunk.token_count_estimate,
            aggregation=str(row.get("aggregation") or "mean_over_chunk"),
            current_em_state=current,
            neutral_baseline_state=neutral,
            prior_current_state=prior,
            delta_vs_neutral=delta_neutral,
            delta_vs_current=delta_current,
            saturation=saturation,
            residual_headroom=residual_headroom,
            positive_mass=positive_mass,
            negative_mass=negative_mass,
            total_mass=total_mass,
            signed_balance=signed_balance,
            sae_feature_values=sae_values,
            sae_delta_vs_neutral=sae_delta_neutral,
            sae_delta_vs_current=sae_delta_current,
            sae_feature_mask=sae_mask,
            sae_novelty=_optional_float(row.get("sae_novelty")),
            sae_overlap_with_memory=_optional_float(row.get("sae_overlap_with_memory")),
            telemetry_valid=bool(row.get("telemetry_valid", True)),
            invalid_reason=row.get("invalid_reason"),
            provenance=provenance,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_valid_prefill_row(row: dict[str, Any]) -> bool:
    if row.get("telemetry_valid") is False:
        return False
    if row.get("invalid_reason"):
        return False
    return bool(_float_map(row.get("sae_feature_values")))


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if raw is not None}


def _bool_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(raw) for key, raw in value.items()}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
