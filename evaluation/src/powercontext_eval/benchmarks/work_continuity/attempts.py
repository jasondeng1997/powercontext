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

"""Recorded continuation attempts as a scored input artifact.

The harness never runs a model. A host integration records what a fresh session
did with the continuation context it received, and this module validates that
recording before it is scored. A recorder maps each step onto the task's declared
fact ids, so scoring stays exact instead of depending on text matching against
free-form model output.

Validation is fail-closed: an attempt that names an undeclared fact id, an
unknown task or arm, or a malformed step is rejected rather than scored, because
a silently dropped reference would understate every conflict metric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from powercontext_eval.benchmarks.work_continuity.arms import ContinuationArmError, get_continuation_arm
from powercontext_eval.benchmarks.work_continuity.catalog import TaskCatalog, WorkContinuityCatalogError
from powercontext_eval.errors import PowerContextEvalError

ATTEMPTS_SCHEMA = "powercontext.work-continuity-attempts.v1"


class AttemptInputError(PowerContextEvalError):
    """A recorded continuation attempt cannot be trusted."""


class AttemptEnvironmentError(AttemptInputError):
    """The recorded attempt artifact or its environment is not ready."""


@dataclass(frozen=True)
class RecordedStep:
    """One action a fresh session took, with the declared facts it relied on."""

    step: int
    action_text: str
    relied_on: tuple[str, ...]
    correction: bool


@dataclass(frozen=True)
class RecordedAttempt:
    """One recorded continuation attempt for one task under one method."""

    task_id: str
    arm_id: str
    host: str
    host_revision: str | None
    model: str | None
    steps: tuple[RecordedStep, ...]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.task_id, self.arm_id, self.host)


@dataclass(frozen=True)
class AttemptSet:
    """Validated recorded attempts for one task catalog."""

    path: Path
    content_sha256: str
    attempts: tuple[RecordedAttempt, ...]

    def for_task_and_arm(self, task_id: str, arm_id: str) -> tuple[RecordedAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.task_id == task_id and attempt.arm_id == arm_id)

    @property
    def hosts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(attempt.host for attempt in self.attempts))

    @property
    def keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(attempt.key for attempt in self.attempts)


def load_attempts(path: Path, *, catalog: TaskCatalog) -> AttemptSet:
    """Read and validate a recorded-attempt artifact against one task catalog."""

    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise AttemptEnvironmentError(f"Cannot read work-continuity attempts: {resolved}") from error
    if not raw.strip():
        raise AttemptEnvironmentError(f"Work-continuity attempts are blank: {resolved}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttemptInputError("Work-continuity attempts are not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {"schema", "attempts"}:
        raise AttemptInputError("Work-continuity attempts must contain only schema and attempts")
    if value["schema"] != ATTEMPTS_SCHEMA:
        raise AttemptInputError("Work-continuity attempts schema is unsupported")
    raw_attempts = value["attempts"]
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise AttemptInputError("Work-continuity attempts must contain at least one attempt")
    attempts: list[RecordedAttempt] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw_attempt in enumerate(raw_attempts):
        attempt = _attempt(raw_attempt, index, catalog)
        if attempt.key in seen:
            task_id, arm_id, host = attempt.key
            raise AttemptInputError(f"Work-continuity attempts repeat task {task_id} arm {arm_id} host {host!r}")
        seen.add(attempt.key)
        attempts.append(attempt)
    return AttemptSet(
        path=resolved,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        attempts=tuple(attempts),
    )


def _mapping(raw: object, error: str) -> dict[str, object]:
    """Return ``raw`` as a JSON object mapping, or reject it with ``error``."""

    if not isinstance(raw, dict):
        raise AttemptInputError(error)
    return cast("dict[str, object]", raw)


def _attempt(raw: object, index: int, catalog: TaskCatalog) -> RecordedAttempt:
    label = f"attempt {index}"
    raw_attempt = _mapping(raw, f"Work-continuity {label} must be an object")
    required = {"task_id", "arm_id", "host", "steps"}
    optional = {"host_revision", "model"}
    keys = set(raw_attempt)
    if not required <= keys or keys - required - optional:
        raise AttemptInputError(
            f"Work-continuity {label} must declare {sorted(required)} and may add {sorted(optional)}"
        )
    try:
        task = catalog.require(_text(raw_attempt["task_id"], f"{label} task_id"))
    except WorkContinuityCatalogError as error:
        raise AttemptInputError(str(error)) from None
    arm_id = _text(raw_attempt["arm_id"], f"{label} arm_id")
    try:
        get_continuation_arm(arm_id)
    except ContinuationArmError as error:
        raise AttemptInputError(str(error)) from None
    declared = set(task.required_fact_ids) | set(task.obsolete_fact_ids)
    steps = _steps(raw_attempt["steps"], label, task.task_id, declared)
    return RecordedAttempt(
        task_id=task.task_id,
        arm_id=arm_id,
        host=_text(raw_attempt["host"], f"{label} host"),
        host_revision=_optional_text(raw_attempt.get("host_revision"), f"{label} host_revision"),
        model=_optional_text(raw_attempt.get("model"), f"{label} model"),
        steps=steps,
    )


def _steps(raw: object, label: str, task_id: str, declared: set[str]) -> tuple[RecordedStep, ...]:
    if not isinstance(raw, list) or not raw:
        raise AttemptInputError(f"Work-continuity {label} must record at least one step")
    steps: list[RecordedStep] = []
    for index, raw_step in enumerate(raw):
        step = _mapping(raw_step, f"Work-continuity {label} step {index} must be an object")
        keys = set(step)
        if not {"step", "action_text", "relied_on"} <= keys or keys - {
            "step",
            "action_text",
            "relied_on",
            "correction",
        }:
            raise AttemptInputError(
                f"Work-continuity {label} step {index} must declare step, action_text, and relied_on, "
                "and may add correction"
            )
        number = step["step"]
        if not isinstance(number, int) or isinstance(number, bool) or number != index + 1:
            raise AttemptInputError(f"Work-continuity {label} steps must be numbered 1..n in order")
        relied_on = step["relied_on"]
        if not isinstance(relied_on, list):
            raise AttemptInputError(f"Work-continuity {label} step {number} relied_on must be an array")
        references: list[str] = []
        for entry in relied_on:
            fact_id = _text(entry, f"{label} step {number} relied_on")
            if fact_id not in declared:
                raise AttemptInputError(
                    f"Work-continuity {label} step {number} relies on undeclared fact {fact_id!r} for task {task_id}"
                )
            if fact_id in references:
                raise AttemptInputError(f"Work-continuity {label} step {number} repeats fact {fact_id}")
            references.append(fact_id)
        correction = step.get("correction", False)
        if not isinstance(correction, bool):
            raise AttemptInputError(f"Work-continuity {label} step {number} correction must be a boolean")
        steps.append(
            RecordedStep(
                step=number,
                action_text=_text(step["action_text"], f"{label} step {number} action_text"),
                relied_on=tuple(references),
                correction=correction,
            )
        )
    return tuple(steps)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttemptInputError(f"Work-continuity {label} must be a non-empty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)
