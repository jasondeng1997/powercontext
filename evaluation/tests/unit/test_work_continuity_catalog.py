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

import json
from pathlib import Path

import pytest
from work_continuity_fixtures import (
    coding_task,
    documentation_task,
    shipped_locks_dir,
    standard_lock,
    valid_attempt,
    write_attempts,
    write_lock,
)

from powercontext_eval.benchmarks.work_continuity.attempts import AttemptInputError, load_attempts
from powercontext_eval.benchmarks.work_continuity.catalog import (
    TaskCatalog,
    WorkContinuityEnvironmentError,
    WorkContinuityInputError,
)


def test_catalog_loads_both_tasks_and_fixes_their_evidence(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))

    assert catalog.task_ids == ("t-coding", "t-doc")
    assert catalog.task_set_id == "test-set"
    assert len(catalog.content_sha256) == 64
    coding = catalog.require("t-coding")
    assert coding.turn_numbers == (1, 2, 3, 4, 5)
    assert coding.obsolete_source_turns == (2,)
    assert coding.required_fact_ids == ("f1", "f2")
    doc = catalog.require("t-doc")
    assert doc.unavailable_fact_ids == ("g3",)
    assert doc.known_omissions == ("The docs build was never run.",)


def test_catalog_rejects_a_lock_that_is_not_the_pinned_schema(tmp_path: Path) -> None:
    path = write_lock(tmp_path, [coding_task(), documentation_task()], schema="other")
    with pytest.raises(WorkContinuityInputError, match="schema is unsupported"):
        TaskCatalog.load(path)


def test_catalog_rejects_a_domain_error_field_in_the_lock(tmp_path: Path) -> None:
    task = coding_task()
    task["unexpected"] = True
    with pytest.raises(WorkContinuityInputError, match="must declare exactly"):
        TaskCatalog.load(write_lock(tmp_path, [task, documentation_task()]))


def test_catalog_rejects_a_transcript_whose_turns_are_not_numbered_in_order(tmp_path: Path) -> None:
    task = coding_task()
    task["transcript"] = [
        {"turn": 1, "role": "user", "text": "a"},
        {"turn": 3, "role": "assistant", "text": "b"},
        {"turn": 4, "role": "assistant", "text": "c"},
        {"turn": 5, "role": "assistant", "text": "d"},
    ]
    with pytest.raises(WorkContinuityInputError, match="numbered 1..n in order"):
        TaskCatalog.load(write_lock(tmp_path, [task, documentation_task()]))


def test_catalog_rejects_evidence_pointing_at_a_turn_that_does_not_exist(tmp_path: Path) -> None:
    task = coding_task()
    task["required_state_facts"] = [
        {"fact_id": "f1", "text": "x", "evidence": "turn:99"},
        {"fact_id": "f2", "text": "y", "evidence": "turn:5"},
    ]
    with pytest.raises(WorkContinuityInputError, match="turn:N pointer"):
        TaskCatalog.load(write_lock(tmp_path, [task, documentation_task()]))


def test_catalog_rejects_a_next_action_that_depends_on_an_undeclared_fact(tmp_path: Path) -> None:
    task = coding_task()
    task["expected_next_action"] = {"action_id": "a1", "text": "do it", "required_fact_ids": ["f1", "f9"]}
    with pytest.raises(WorkContinuityInputError, match="unknown state fact f9"):
        TaskCatalog.load(write_lock(tmp_path, [task, documentation_task()]))


def test_catalog_rejects_an_obsolete_fact_that_reuses_a_required_fact_id(tmp_path: Path) -> None:
    task = coding_task()
    task["obsolete_facts"] = [{"fact_id": "f1", "text": "stale", "evidence": "turn:2", "superseded_by": "f1"}]
    with pytest.raises(WorkContinuityInputError, match="reuses a required state fact id"):
        TaskCatalog.load(write_lock(tmp_path, [task, documentation_task()]))


def test_catalog_rejects_an_evidence_free_fact_that_is_not_declared_unavailable(tmp_path: Path) -> None:
    task = documentation_task(evidence=None)
    task["unavailable_evidence"] = []
    with pytest.raises(WorkContinuityInputError, match="must match its evidence-free facts exactly"):
        TaskCatalog.load(write_lock(tmp_path, [task, coding_task()]))


def test_catalog_rejects_an_unavailable_entry_for_a_turn_backed_fact(tmp_path: Path) -> None:
    task = documentation_task(evidence="turn:3")
    with pytest.raises(WorkContinuityInputError, match="declared_but_turn_backed=g3"):
        TaskCatalog.load(write_lock(tmp_path, [task, coding_task()]))


def test_catalog_rejects_a_task_set_without_both_task_kinds(tmp_path: Path) -> None:
    second = coding_task()
    second["task_id"] = "t-coding-2"
    with pytest.raises(WorkContinuityInputError, match="must cover both coding and documentation"):
        TaskCatalog.load(write_lock(tmp_path, [coding_task(), second]))


def test_catalog_rejects_a_task_set_without_a_superseded_fact(tmp_path: Path) -> None:
    coding = coding_task()
    coding["obsolete_facts"] = []
    with pytest.raises(WorkContinuityInputError, match="at least one superseded state fact"):
        TaskCatalog.load(write_lock(tmp_path, [coding, documentation_task()]))


def test_catalog_rejects_a_task_set_without_an_unavailable_evidence_case(tmp_path: Path) -> None:
    doc = documentation_task()
    doc["required_state_facts"] = [
        {"fact_id": "g1", "text": "commit compares the expected head", "evidence": "turn:2"},
        {"fact_id": "g2", "text": "the change belongs in docs/guide.md", "evidence": "turn:4"},
    ]
    doc["unavailable_evidence"] = []
    with pytest.raises(WorkContinuityInputError, match="at least one unavailable-evidence case"):
        TaskCatalog.load(write_lock(tmp_path, [coding_task(), doc]))


def test_catalog_require_and_select_report_declared_identifiers(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))

    assert [task.task_id for task in catalog.select(None)] == ["t-coding", "t-doc"]
    assert [task.task_id for task in catalog.select(["t-doc"])] == ["t-doc"]
    with pytest.raises(WorkContinuityInputError, match="unknown work-continuity task"):
        catalog.require("missing")
    with pytest.raises(WorkContinuityInputError, match="Duplicate work-continuity task selection"):
        catalog.select(["t-doc", "t-doc"])


def test_catalog_reports_an_unreadable_lock_as_an_environment_error(tmp_path: Path) -> None:
    with pytest.raises(WorkContinuityEnvironmentError, match="Cannot read"):
        TaskCatalog.load(tmp_path / "absent.json")


def test_catalog_rejects_a_blank_lock(tmp_path: Path) -> None:
    path = tmp_path / "blank.json"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(WorkContinuityEnvironmentError, match="is blank"):
        TaskCatalog.load(path)


def test_attempts_load_and_group_by_host(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    path = write_attempts(
        tmp_path,
        [valid_attempt(), valid_attempt(host="second-host"), valid_attempt(arm_id="full-transcript-v1")],
    )

    attempts = load_attempts(path, catalog=catalog)

    assert attempts.hosts == ("fixture-host", "second-host")
    assert len(attempts.for_task_and_arm("t-coding", "full-transcript-v1")) == 1
    assert attempts.for_task_and_arm("t-coding", "informal-summary-v1") == ()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (valid_attempt(task_id="nope"), "unknown work-continuity task"),
        (valid_attempt(arm_id="not-an-arm"), "unknown continuation arm"),
        (valid_attempt(host="   "), "host must be a non-empty string"),
        (valid_attempt(steps=[]), "must record at least one step"),
    ],
)
def test_attempts_reject_entries_that_cannot_be_scored(tmp_path: Path, entry: dict[str, object], message: str) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    with pytest.raises(AttemptInputError, match=message):
        load_attempts(write_attempts(tmp_path, [entry]), catalog=catalog)


def test_attempts_reject_an_undeclared_fact_reference(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    entry = valid_attempt(steps=[{"step": 1, "action_text": "x", "relied_on": ["f1", "f9"]}])
    with pytest.raises(AttemptInputError, match="relies on undeclared fact 'f9'"):
        load_attempts(write_attempts(tmp_path, [entry]), catalog=catalog)


def test_attempts_reject_an_out_of_order_step_number(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    entry = valid_attempt(
        steps=[
            {"step": 1, "action_text": "x", "relied_on": []},
            {"step": 3, "action_text": "y", "relied_on": []},
        ]
    )
    with pytest.raises(AttemptInputError, match="numbered 1..n in order"):
        load_attempts(write_attempts(tmp_path, [entry]), catalog=catalog)


def test_attempts_reject_a_duplicate_task_arm_host_triple(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    with pytest.raises(AttemptInputError, match="repeat task t-coding arm rollover-handoff-v1"):
        load_attempts(write_attempts(tmp_path, [valid_attempt(), valid_attempt()]), catalog=catalog)


def test_attempts_reject_an_unknown_schema_and_accept_the_optional_identity_fields(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    path = tmp_path / "attempts.json"
    path.write_text(
        json.dumps({"schema": "powercontext.work-continuity-attempts.v9", "attempts": [valid_attempt()]}),
        encoding="utf-8",
    )
    with pytest.raises(AttemptInputError, match="schema is unsupported"):
        load_attempts(path, catalog=catalog)

    accepted = load_attempts(write_attempts(tmp_path, [valid_attempt(model="m", host_revision="r")]), catalog=catalog)
    assert accepted.attempts[0].model == "m"
    assert accepted.attempts[0].host_revision == "r"


def test_attempts_reject_a_step_without_a_non_empty_action_text(tmp_path: Path) -> None:
    catalog = TaskCatalog.load(standard_lock(tmp_path))
    entry = valid_attempt(steps=[{"step": 1, "action_text": "  ", "relied_on": []}])
    with pytest.raises(AttemptInputError, match="action_text must be a non-empty string"):
        load_attempts(write_attempts(tmp_path, [entry]), catalog=catalog)


def test_shipped_fixture_lock_and_attempts_validate() -> None:
    """The checked-in fixture must stay loadable and internally consistent."""

    locks = shipped_locks_dir()
    catalog = TaskCatalog.load(locks / "work-continuity-v1.tasks.json")
    attempts = load_attempts(locks / "work-continuity-v1.attempts-fixture.json", catalog=catalog)

    assert len(catalog.tasks) >= 6
    assert any(task.unavailable_evidence for task in catalog.tasks.values())
    assert any(task.obsolete_facts for task in catalog.tasks.values())
    assert len(attempts.attempts) == len(catalog.tasks) * 4 * len(attempts.hosts)
