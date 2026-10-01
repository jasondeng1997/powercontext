# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestration for one work-continuity run.

The run has two halves that must stay separable: assembly always works and needs
no host, while scoring needs recorded attempts. These tests pin that split, the
manifest a reader reproduces a run from, and the refusals that stop a run from
looking complete when it is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from work_continuity_fixtures import analysis_lock, write_attempts

from powercontext_eval.benchmarks.work_continuity.analysis import FAILURE_CLASSES, NO_RECORDING
from powercontext_eval.benchmarks.work_continuity.arms import (
    COMPACTED_TRANSCRIPT,
    FULL_TRANSCRIPT,
    INFORMAL_SUMMARY,
    ROLLOVER_HANDOFF,
    supported_continuation_arm_ids,
)
from powercontext_eval.benchmarks.work_continuity.catalog import TaskCatalog
from powercontext_eval.benchmarks.work_continuity.runner import (
    ASSEMBLY_ONLY_CLASSIFICATION,
    RECORDED_CLASSIFICATION,
    RUN_MANIFEST_SCHEMA,
    WorkContinuityRun,
    WorkContinuityRunError,
    run_summary,
    run_work_continuity,
    write_run_artifacts,
)

HOST = "fixture-host"
ARMS = (FULL_TRANSCRIPT, COMPACTED_TRANSCRIPT, INFORMAL_SUMMARY, ROLLOVER_HANDOFF)


def step(number: int, relied_on: list[str]) -> dict[str, Any]:
    return {"step": number, "action_text": f"action {number}", "relied_on": relied_on}


def attempt(task_id: str, arm_id: str, *steps: dict[str, Any], host: str = HOST) -> dict[str, Any]:
    return {"task_id": task_id, "arm_id": arm_id, "host": host, "steps": list(steps)}


def attempts_entries() -> list[dict[str, Any]]:
    """Cover every arm on both tasks, with one deliberate treatment failure."""

    return [
        attempt("t-audit", FULL_TRANSCRIPT.arm_id, step(1, ["h1", "h2"])),
        attempt("t-audit", COMPACTED_TRANSCRIPT.arm_id, step(1, ["h1", "h2"])),
        attempt("t-audit", INFORMAL_SUMMARY.arm_id, step(1, ["h2"])),
        attempt("t-audit", ROLLOVER_HANDOFF.arm_id, step(1, ["h1"])),
        attempt("t-doc", FULL_TRANSCRIPT.arm_id, step(1, ["g1", "g2"])),
        attempt("t-doc", COMPACTED_TRANSCRIPT.arm_id, step(1, ["g1", "g2"])),
        attempt("t-doc", INFORMAL_SUMMARY.arm_id, step(1, ["g2"])),
        attempt("t-doc", ROLLOVER_HANDOFF.arm_id, step(1, ["g1", "g2"])),
    ]


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    return analysis_lock(tmp_path)


@pytest.fixture
def attempts(tmp_path: Path) -> Path:
    return write_attempts(tmp_path, attempts_entries())


def recorded_run(lock: Path, attempts: Path) -> WorkContinuityRun:
    return run_work_continuity(task_lock=lock, run_id="run-1", attempts_path=attempts)


def summary_arms(run: WorkContinuityRun) -> dict[str, dict[str, Any]]:
    """Return the per-arm summary blocks keyed by arm id."""

    arms = run_summary(run)["arms"]
    assert isinstance(arms, list)
    entries: dict[str, dict[str, Any]] = {}
    for entry in arms:
        assert isinstance(entry, dict)
        arm_id = entry["arm_id"]
        assert isinstance(arm_id, str)
        entries[arm_id] = entry
    return entries


def nested(block: dict[str, Any], key: str) -> dict[str, Any]:
    value = block[key]
    assert isinstance(value, dict)
    return value


def write_isolated_attempts(tmp_path: Path, name: str, entries: list[dict[str, Any]]) -> Path:
    """Write an attempts artifact into its own directory so fixtures never collide."""

    directory = tmp_path / name
    directory.mkdir()
    return write_attempts(directory, entries)


def test_an_assembly_only_run_needs_no_attempts(lock: Path) -> None:
    run = run_work_continuity(task_lock=lock, run_id="assembly-only")

    assert run.classification == ASSEMBLY_ONLY_CLASSIFICATION
    assert run.analysis is None
    assert run.scores == ()
    assert run.hosts == ()
    assert len(run.contexts) == len(run.tasks) * len(supported_continuation_arm_ids())


def test_a_recorded_run_classifies_itself_as_incomplete_on_purpose(lock: Path, attempts: Path) -> None:
    run = recorded_run(lock, attempts)

    assert run.classification == RECORDED_CLASSIFICATION
    assert run.classification != "complete-benchmark-result"
    assert run.analysis is not None
    assert run.hosts == (HOST,)
    assert len(run.scores) == len(attempts_entries())


def test_the_manifest_pins_every_input_a_reader_needs_to_reproduce_the_run(lock: Path, attempts: Path) -> None:
    manifest = recorded_run(lock, attempts).manifest
    catalog = TaskCatalog.load(lock)

    assert manifest["schema"] == RUN_MANIFEST_SCHEMA
    assert manifest["workload"] == "work-continuity"
    assert manifest["run_id"] == "run-1"
    assert manifest["task_set_id"] == catalog.task_set_id
    assert manifest["task_ids"] == ["t-audit", "t-doc"]
    assert manifest["assembly"] == {"max_bytes": 16_000}
    assert manifest["hosts"] == [HOST]
    assert len(str(manifest["task_evidence_digest"])) == 64

    inputs = nested(manifest, "inputs")
    assert nested(inputs, "task_lock")["content_sha256"] == catalog.content_sha256
    assert nested(inputs, "attempts")["path"] == attempts.name

    comparable = manifest["comparable_arms"]
    assert isinstance(comparable, list)
    assert [record["id"] for record in comparable] == [arm.arm_id for arm in ARMS]


def test_the_manifest_names_the_ground_truth_it_scored_against(lock: Path, attempts: Path, tmp_path: Path) -> None:
    whole = recorded_run(lock, attempts).manifest["task_evidence_digest"]
    narrow = write_isolated_attempts(
        tmp_path, "ground-truth", [attempt("t-audit", FULL_TRANSCRIPT.arm_id, step(1, ["h1", "h2"]))]
    )

    narrowed = run_work_continuity(
        task_lock=lock, run_id="run-2", task_ids=("t-audit",), attempts_path=narrow
    ).manifest["task_evidence_digest"]

    assert whole != narrowed


def test_the_run_summary_keeps_injected_bytes_and_outcome_in_separate_blocks(lock: Path, attempts: Path) -> None:
    for entry in summary_arms(recorded_run(lock, attempts)).values():
        injected = nested(entry, "injected_bytes")
        outcome = nested(entry, "outcome")

        assert injected["total"] > 0
        assert set(outcome) == {
            "recorded_attempts",
            "task_success",
            "incorrect_assumptions",
            "missing_evidence",
            "unverifiable_claims",
            "user_correction_burden",
            "mean_time_to_recover_state",
        }
        # No outcome metric is derived from a byte count.
        assert "injected_bytes" not in outcome


def test_the_run_reports_the_treatment_as_underperforming_on_exactly_one_task(lock: Path, attempts: Path) -> None:
    run = recorded_run(lock, attempts)
    assert run.analysis is not None

    assert run.analysis.underperforming_task_ids == ("t-audit",)
    entry = run.analysis.underperformance[0]
    assert (entry.task_id, entry.host) == ("t-audit", HOST)
    assert entry.beaten_by == (FULL_TRANSCRIPT.arm_id, COMPACTED_TRANSCRIPT.arm_id)
    assert entry.treatment_failure_class == "vague_next_action"


def test_an_assembly_only_run_has_no_failure_block(lock: Path) -> None:
    run = run_work_continuity(task_lock=lock, run_id="assembly-only")
    summary = run_summary(run)

    assert "failure_analysis" not in summary
    for entry in summary_arms(run).values():
        assert nested(entry, "outcome")["recorded_attempts"] == 0


def test_a_smaller_context_is_not_thereby_a_better_continuation(lock: Path, attempts: Path) -> None:
    """The handoff injects fewer bytes than the full transcript and still loses on t-audit."""

    arms = summary_arms(recorded_run(lock, attempts))
    handoff = arms[ROLLOVER_HANDOFF.arm_id]
    full = arms[FULL_TRANSCRIPT.arm_id]

    assert nested(handoff, "injected_bytes")["total"] < nested(full, "injected_bytes")["total"]
    assert nested(handoff, "outcome")["task_success"] < nested(full, "outcome")["task_success"]


def test_run_artifacts_are_written_into_a_fresh_directory(lock: Path, attempts: Path, tmp_path: Path) -> None:
    run = recorded_run(lock, attempts)
    target = tmp_path / "run-out"

    artifacts = write_run_artifacts(run, target)

    assert artifacts.output_dir == target
    for path in (artifacts.manifest_path, artifacts.assembly_path, artifacts.scores_path, artifacts.summary_path):
        assert path.exists() and path.stat().st_size > 0

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == RUN_MANIFEST_SCHEMA
    rows = [json.loads(line) for line in artifacts.assembly_path.read_text(encoding="utf-8").splitlines()]
    assert {row["arm_id"] for row in rows} == set(supported_continuation_arm_ids())
    scored = [json.loads(line) for line in artifacts.scores_path.read_text(encoding="utf-8").splitlines()]
    assert len(scored) == len(attempts_entries())


def test_writing_run_artifacts_refuses_to_overwrite_an_existing_directory(
    lock: Path, attempts: Path, tmp_path: Path
) -> None:
    run = recorded_run(lock, attempts)
    target = tmp_path / "run-out"
    write_run_artifacts(run, target)

    with pytest.raises(WorkContinuityRunError, match="output directory already exists"):
        write_run_artifacts(run, target)


def test_a_run_refuses_attempts_that_name_a_task_it_did_not_select(lock: Path, attempts: Path) -> None:
    with pytest.raises(WorkContinuityRunError, match=r"attempts name task t-doc but the run selects \['t-audit'\]"):
        run_work_continuity(task_lock=lock, run_id="narrow", task_ids=("t-audit",), attempts_path=attempts)


def test_a_run_refuses_attempts_that_name_an_arm_it_did_not_select(lock: Path, attempts: Path) -> None:
    with pytest.raises(
        WorkContinuityRunError,
        match=r"attempts name arm compacted-transcript-v1 but the run selects \['full-transcript-v1'\]",
    ):
        run_work_continuity(
            task_lock=lock,
            run_id="narrow",
            arm_ids=(FULL_TRANSCRIPT.arm_id,),
            attempts_path=attempts,
        )


def test_an_unrecorded_combination_is_a_recording_gap_rather_than_a_failure(lock: Path, tmp_path: Path) -> None:
    partial = write_attempts(tmp_path, [attempt("t-audit", FULL_TRANSCRIPT.arm_id, step(1, ["h1", "h2"]))])

    run = run_work_continuity(task_lock=lock, run_id="partial", attempts_path=partial)
    assert run.analysis is not None

    assert run.analysis.unrecorded_keys
    assert {finding.failure_class for finding in run.analysis.findings} == {NO_RECORDING}
    # A recording gap is deliberately kept out of the failure-class histogram, so
    # an incomplete run never reads as a finding about a method.
    assert set(run.analysis.findings_by_class) == set(FAILURE_CLASSES)
    assert sum(run.analysis.findings_by_class.values()) == 0


def test_selecting_one_arm_and_one_task_narrows_the_run(lock: Path, tmp_path: Path) -> None:
    narrow = write_isolated_attempts(
        tmp_path, "narrow", [attempt("t-audit", FULL_TRANSCRIPT.arm_id, step(1, ["h1", "h2"]))]
    )

    run = run_work_continuity(
        task_lock=lock,
        run_id="narrow",
        arm_ids=(FULL_TRANSCRIPT.arm_id,),
        task_ids=("t-audit",),
        attempts_path=narrow,
    )

    assert [context.task_id for context in run.contexts] == ["t-audit"]
    assert [context.arm_id for context in run.contexts] == [FULL_TRANSCRIPT.arm_id]


@pytest.mark.parametrize("max_bytes", [0, -5])
def test_a_non_positive_budget_is_refused(lock: Path, max_bytes: int) -> None:
    with pytest.raises(WorkContinuityRunError, match="assembly budget must be positive"):
        run_work_continuity(task_lock=lock, run_id="bad", max_bytes=max_bytes)


def test_the_context_and_task_accessors_report_unknown_identifiers(lock: Path) -> None:
    run = run_work_continuity(task_lock=lock, run_id="assembly-only")

    with pytest.raises(WorkContinuityRunError, match="no assembled context for task t-audit arm nope"):
        run.context("t-audit", "nope")
    with pytest.raises(WorkContinuityRunError, match="no selected task nope"):
        run.task("nope")


def test_revisions_are_recorded_verbatim_when_supplied(lock: Path) -> None:
    run = run_work_continuity(
        task_lock=lock,
        run_id="revisions",
        powercontext_revision="pc-abc",
        integration_revision="int-def",
    )

    assert run.manifest["revisions"] == {"powercontext": "pc-abc", "integration": "int-def"}
