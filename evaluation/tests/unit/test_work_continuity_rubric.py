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

"""The work-continuity rubric.

Every metric is derived from the assembled context and the declared facts of a
recorded step, so these tests drive the score from a hand-built attempt rather
than from text matching. The one property that matters most is also the one the
issue calls out: injected bytes stay out of the outcome ranking.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from work_continuity_fixtures import analysis_lock

from powercontext_eval.benchmarks.work_continuity.arms import (
    COMPACTED_TRANSCRIPT,
    FULL_TRANSCRIPT,
    INFORMAL_SUMMARY,
    ROLLOVER_HANDOFF,
    ContinuationArm,
)
from powercontext_eval.benchmarks.work_continuity.assembly import ContinuationContext, assemble_context
from powercontext_eval.benchmarks.work_continuity.attempts import RecordedAttempt, RecordedStep
from powercontext_eval.benchmarks.work_continuity.catalog import TaskCatalog
from powercontext_eval.benchmarks.work_continuity.rubric import (
    UNRECOVERED_STEP_SENTINEL,
    AttemptScore,
    score_attempt,
    score_attempts,
)

GENEROUS_BUDGET = 16_000


@pytest.fixture
def catalog(tmp_path: Path) -> TaskCatalog:
    return TaskCatalog.load(analysis_lock(tmp_path))


def context_for(catalog: TaskCatalog, task_id: str, arm: ContinuationArm) -> ContinuationContext:
    return assemble_context(catalog.require(task_id), arm, max_bytes=GENEROUS_BUDGET)


def recorded(
    *steps: dict[str, Any],
    task_id: str = "t-audit",
    arm_id: str = "full-transcript-v1",
    host: str = "host-a",
) -> RecordedAttempt:
    return RecordedAttempt(
        task_id=task_id,
        arm_id=arm_id,
        host=host,
        host_revision=None,
        model=None,
        steps=tuple(RecordedStep(**step) for step in steps),
    )


def step(number: int, relied_on: list[str], *, correction: bool = False) -> dict[str, Any]:
    return {"step": number, "action_text": f"action {number}", "relied_on": relied_on, "correction": correction}


def test_a_recovered_attempt_reports_success_and_the_step_it_recovered_at(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["h1"]), step(2, ["h1", "h2"]))

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.task_success is True
    assert score.time_to_recover_state == 2
    assert (score.incorrect_assumptions, score.missing_evidence, score.unverifiable_claims) == (0, 0, 0)
    assert score.user_correction_burden == 0
    assert [scored.is_recovery for scored in score.steps] == [False, True]


def test_a_step_only_recovers_when_it_relies_on_every_required_fact(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["h1"]), step(2, ["h2"]), step(3, ["h1", "h2"]))

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.time_to_recover_state == 3


def test_an_attempt_that_never_recovers_reports_no_recovery_step(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["h1"]), step(2, ["h1"]))

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.task_success is False
    assert score.time_to_recover_state is None
    assert score.outcome_rank[0] == 0
    assert score.outcome_rank[-1] == -UNRECOVERED_STEP_SENTINEL


def test_a_superseded_reliance_counts_as_an_incorrect_assumption(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["o1"]), step(2, ["h1", "h2"]))

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.incorrect_assumptions == 1
    assert score.steps[0].superseded_reliance == ("o1",)
    assert score.steps[1].superseded_reliance == ()


def test_a_reliance_on_a_fact_the_context_never_delivered_is_missing_evidence(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", COMPACTED_TRANSCRIPT)
    attempt = recorded(step(1, ["h1", "h3"]), step(2, ["h1", "h2"]), arm_id="compacted-transcript-v1")

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert context.delivered_fact_ids == ("h1", "h2")
    assert score.missing_evidence == 1
    assert score.steps[0].undelivered_reliance == ("h3",)
    assert score.steps[1].undelivered_reliance == ()
    # The action's own facts did arrive, so this is a real evidence conflict and
    # not an artifact of the assembly.
    assert score.has_assembly_gap is False


def test_a_reliance_on_unavailable_evidence_is_an_unverifiable_claim(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-doc", ROLLOVER_HANDOFF)
    attempt = recorded(
        step(1, ["g3"]),
        step(2, ["g1", "g2"]),
        task_id="t-doc",
        arm_id="rollover-handoff-v1",
    )

    score = score_attempt(catalog.require("t-doc"), context, attempt)

    assert context.unavailable_fact_ids == ("g3",)
    assert score.unverifiable_claims == 1
    assert score.steps[0].unavailable_reliance == ("g3",)
    assert score.missing_evidence == 0


def test_a_recorded_correction_counts_toward_the_user_correction_burden(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["h1"], correction=True), step(2, ["h1", "h2"]))

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.user_correction_burden == 1
    assert score.steps[0].correction is True


def test_the_score_carries_the_contexts_injected_bytes_verbatim(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", ROLLOVER_HANDOFF)
    attempt = recorded(step(1, ["h1", "h2"]), arm_id="rollover-handoff-v1")

    score = score_attempt(catalog.require("t-audit"), context, attempt)

    assert score.injected_bytes == context.injected_bytes
    assert score.max_bytes == GENEROUS_BUDGET
    assert score.context_truncated is False


def test_the_outcome_rank_deliberately_excludes_injected_bytes(catalog: TaskCatalog) -> None:
    """Injecting fewer bytes must never improve the continuation ranking."""

    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempt = recorded(step(1, ["h1", "h2"]))
    score = score_attempt(catalog.require("t-audit"), context, attempt)

    cheaper = replace(score, injected_bytes=score.injected_bytes // 10, max_bytes=64)
    costlier = replace(score, injected_bytes=score.injected_bytes * 10, max_bytes=1_000_000)

    assert cheaper.outcome_rank == score.outcome_rank == costlier.outcome_rank
    assert len(score.outcome_rank) == 6


def test_a_recovered_attempt_always_ranks_above_an_unrecovered_one(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    task = catalog.require("t-audit")

    recovered = score_attempt(task, context, recorded(step(1, ["h1", "h2"])))
    unrecovered = score_attempt(task, context, recorded(step(1, ["h1"])))

    assert unrecovered.outcome_rank < recovered.outcome_rank


def test_conflicts_rank_below_a_clean_recovery_with_the_same_success(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    task = catalog.require("t-audit")

    clean = score_attempt(task, context, recorded(step(1, ["h1", "h2"])))
    stale = score_attempt(task, context, recorded(step(1, ["o1", "h1", "h2"])))

    assert clean.task_success is True
    assert stale.task_success is True
    assert stale.outcome_rank < clean.outcome_rank


def test_facts_missing_from_context_names_what_the_action_needed(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", COMPACTED_TRANSCRIPT)
    score = score_attempt(
        catalog.require("t-audit"),
        context,
        recorded(step(1, ["h1", "h2"]), arm_id="compacted-transcript-v1"),
    )

    assert score.expected_fact_ids == ("h1", "h2")
    assert score.facts_missing_from_context == ()
    assert score.has_assembly_gap is False


def test_a_context_that_dropped_a_required_fact_reports_an_assembly_gap(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", INFORMAL_SUMMARY)
    score = score_attempt(catalog.require("t-audit"), context, recorded(step(1, ["h2"]), arm_id="informal-summary-v1"))

    assert score.delivered_fact_ids == ("h2",)
    assert score.facts_missing_from_context == ("h1",)
    assert score.has_assembly_gap is True


def test_scoring_refuses_an_attempt_from_a_different_task_or_arm(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    foreign_task = recorded(step(1, ["h1"]), task_id="t-doc", arm_id="full-transcript-v1")

    with pytest.raises(ValueError, match="attempt does not belong to this task and arm"):
        score_attempt(catalog.require("t-audit"), context, foreign_task)


def test_scoring_every_attempt_for_one_task_and_arm_keeps_them_separate(catalog: TaskCatalog) -> None:
    context = context_for(catalog, "t-audit", FULL_TRANSCRIPT)
    attempts = (
        recorded(step(1, ["h1", "h2"]), host="host-a"),
        recorded(step(1, ["h1"]), step(2, ["h1", "h2"]), host="host-b"),
    )

    scores = score_attempts(catalog.require("t-audit"), context, attempts)

    assert [score.host for score in scores] == ["host-a", "host-b"]
    assert [score.time_to_recover_state for score in scores] == [1, 2]


def test_two_runs_of_the_same_arm_over_the_same_task_produce_identical_numbers(catalog: TaskCatalog) -> None:
    task = catalog.require("t-audit")
    attempt = recorded(step(1, ["o1"]), step(2, ["h1", "h2"]))

    first = score_attempt(task, context_for(catalog, "t-audit", FULL_TRANSCRIPT), attempt)
    second = score_attempt(task, context_for(catalog, "t-audit", FULL_TRANSCRIPT), attempt)

    assert first == second


def test_a_hand_built_score_without_steps_still_answers_the_gap_properties() -> None:
    """The two derived properties read only the fact id tuples, not the steps."""

    score = AttemptScore(
        task_id="t-audit",
        arm_id="rollover-handoff-v1",
        host="host-a",
        task_success=True,
        time_to_recover_state=1,
        incorrect_assumptions=0,
        missing_evidence=0,
        unverifiable_claims=0,
        user_correction_burden=0,
        injected_bytes=100,
        max_bytes=GENEROUS_BUDGET,
        context_truncated=False,
        delivered_fact_ids=("h1", "h2"),
        expected_fact_ids=("h1", "h2"),
        steps=(),
    )

    assert score.facts_missing_from_context == ()
    assert score.has_assembly_gap is False
