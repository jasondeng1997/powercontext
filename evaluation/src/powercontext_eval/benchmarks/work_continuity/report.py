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

"""Unified report for one work-continuity run.

The report keeps two tables that are never merged. One states what each method
injected; the other states how each method's continuations turned out. Issue
requirement: injected bytes are reported separately from task success, so a
smaller context is never quietly presented as a better continuation.

Every report states its own boundary. A fixture task set and synthetic recorded
attempts validate the harness; they are not evidence about a real host, model, or
PowerContext build, and the report says so in its own first line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from powercontext_eval.benchmarks.work_continuity.analysis import (
    NO_RECORDING,
    ArmOutcome,
    WorkContinuityAnalysis,
)
from powercontext_eval.benchmarks.work_continuity.arms import TREATMENT_ARM_ID
from powercontext_eval.benchmarks.work_continuity.runner import WorkContinuityRun, run_summary

REPORT_SCHEMA = "powercontext.work-continuity-report.v1"

NOT_A_BENCHMARK_BANNER = (
    "Not a complete benchmark result. This run reports what the checked-in fixture task set and the "
    "supplied recorded attempts produced under one declared protocol. It does not establish that "
    "Rollover Handoff outperforms any alternative in production, on a real host, or on a real model."
)


class ReportError(Exception):
    """A work-continuity report cannot be rendered from the supplied run."""


@dataclass(frozen=True)
class ReportResult:
    """Paths written for one work-continuity report."""

    report_path: Path
    markdown_path: Path


def build_report(run: WorkContinuityRun, *, output_dir: Path | None = None) -> ReportResult:
    """Render one run as a machine-readable report plus a human summary."""

    target = (output_dir or Path(".")).resolve()
    target.mkdir(parents=True, exist_ok=True)
    payload = report_payload(run)
    report_path = target / "report.json"
    markdown_path = target / "report.md"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return ReportResult(report_path=report_path, markdown_path=markdown_path)


def report_payload(run: WorkContinuityRun) -> dict[str, object]:
    """Build the unified report body for one run."""

    summary = run_summary(run)
    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "banner": NOT_A_BENCHMARK_BANNER,
        "workload": run.workload,
        "classification": run.classification,
        "run_id": run.run_id,
        "task_set_id": run.task_set_id,
        "assembly_max_bytes": run.max_bytes,
        "hosts": list(run.hosts),
        "separated_measurements": {
            "injected_bytes": "reported per method and per task, never combined with an outcome metric",
            "outcome": "reported per method and per task as success, recovery cost, and conflict counts",
        },
        "methods": summary["arms"],
        "tasks": _task_rows(run),
        "failure_analysis": _failure_block(run.analysis),
        "recommendations": _recommendations(run.analysis),
        "boundaries": _boundaries(run),
    }
    return payload


def render_markdown(payload: dict[str, object]) -> str:
    """Render the unified report body as a human summary."""

    lines: list[str] = [
        f"# Work-continuity report: {payload['run_id']}",
        "",
        f"> {payload['banner']}",
        "",
        f"- Task set: `{payload['task_set_id']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Assembly ceiling: {payload['assembly_max_bytes']} bytes per task",
        f"- Recorded hosts: {', '.join(_host_names(payload))}",
        "",
        "## Injected bytes",
        "",
        "What each method delivered. These numbers are not a success measure.",
        "",
        "| Method | Tasks | Total bytes | Mean bytes | Max bytes | Truncated tasks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _entries(payload["methods"], "methods"):
        injected = _mapping(method.get("injected_bytes"), "methods[].injected_bytes")
        lines.append(
            f"| `{method['arm_id']}` | {method['task_count']} | {injected['total']} | "
            f"{injected['mean']} | {injected['max']} | {injected['truncated_task_count']} |"
        )
    lines.extend(
        [
            "",
            "## Continuation outcome",
            "",
            (
                "What the recorded continuations did. Success counts recovered tasks; the conflict columns count "
                "recorded steps, and none of them is derived from the byte counts above."
            ),
            "",
            (
                "| Method | Recorded | Recovered | Mean recovery step | Incorrect assumptions | "
                "Missing evidence | Unverifiable claims | Corrections |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in _entries(payload["methods"], "methods"):
        outcome = _mapping(method.get("outcome"), "methods[].outcome")
        mean = outcome["mean_time_to_recover_state"]
        lines.append(
            f"| `{method['arm_id']}` | {outcome['recorded_attempts']} | {outcome['task_success']} | "
            f"{'-' if mean is None else mean} | {outcome['incorrect_assumptions']} | "
            f"{outcome['missing_evidence']} | {outcome['unverifiable_claims']} | "
            f"{outcome['user_correction_burden']} |"
        )
    lines.extend(
        [
            "",
            "## Per task",
            "",
            "| Task | Kind | Treatment outcome | Baselines ranked above the treatment |",
            "| --- | --- | --- | --- |",
        ]
    )
    for task in _entries(payload["tasks"], "tasks"):
        by_host = _mapping(task.get("treatment_by_host"), "tasks[].treatment_by_host")
        treatment = (
            "; ".join(f"{host}: {summary}" for host, summary in by_host.items())
            if by_host
            else "assembly only, no recorded attempt"
        )
        beaten_by = task.get("underperforming_baselines")
        if not isinstance(beaten_by, list):
            raise ReportError("report payload tasks[].underperforming_baselines must be an array")
        lines.append(
            f"| `{task['task_id']}` | {task['kind']} | {treatment} | "
            f"{', '.join(str(entry) for entry in beaten_by) if beaten_by else 'none'} |"
        )
    failure = _mapping(payload.get("failure_analysis"), "failure_analysis")
    lines.extend(
        [
            "",
            "## Failure analysis",
            "",
            f"Findings by class: {_inline_counts(failure['findings_by_class'])}",
            "",
        ]
    )
    unrecorded = failure["unrecorded_keys"]
    if not isinstance(unrecorded, list):
        raise ReportError("report payload failure_analysis.unrecorded_keys must be an array")
    lines.append(
        f"Unrecorded task/method/host combinations: {len(unrecorded)}"
        if unrecorded
        else "Every selected task and method has a recorded attempt."
    )
    lines.extend(["", "### Where the treatment underperformed", ""])
    underperformance = _entries(
        failure.get("treatment_underperformance"), "failure_analysis.treatment_underperformance"
    )
    if not underperformance:
        lines.append("The treatment did not rank below a baseline on any host in this run.")
    for entry in underperformance:
        beaten_by = entry.get("beaten_by")
        if not isinstance(beaten_by, list):
            raise ReportError("report payload treatment_underperformance[].beaten_by must be an array")
        caused_by = entry.get("treatment_failure_class") or "no classified failure"
        lines.append(
            f"- `{entry['task_id']}` on `{entry['host']}`: `{TREATMENT_ARM_ID}` ranked below "
            f"{', '.join(f'`{arm}`' for arm in beaten_by)}; treatment failure class "
            f"`{caused_by}`."
        )
    lines.extend(["", "### Findings", ""])
    findings = _entries(failure.get("findings"), "failure_analysis.findings")
    if not findings:
        lines.append("No finding was classified for this run.")
    for finding in findings:
        requirement = finding.get("requirement") or "no contract change requested"
        lines.append(
            f"- `{finding['task_id']}` / `{finding['arm_id']}` / `{finding['host']}`: "
            f"**{finding['failure_class']}** — {finding['detail']} (argues about: {requirement})"
        )
    lines.extend(["", "## Recommendations", ""])
    recommendations = _entries(payload.get("recommendations"), "recommendations")
    if not recommendations:
        lines.append("No contract change is recommended by this run.")
    for entry in recommendations:
        lines.append(
            f"- **{entry['requirement'] or 'recording gap'}** ({entry['failure_class']}, "
            f"{entry['occurrences']} finding(s)): {entry['recommendation']}"
        )
    lines.extend(["", "## Boundaries", ""])
    boundaries = payload["boundaries"]
    if not isinstance(boundaries, list):
        raise ReportError("report payload boundaries must be an array")
    lines.extend(f"- {boundary}" for boundary in boundaries)
    lines.append("")
    return "\n".join(lines)


def _task_rows(run: WorkContinuityRun) -> list[dict[str, object]]:
    if run.analysis is None:
        rows: list[dict[str, object]] = []
        for task in run.tasks:
            contexts = [context for context in run.contexts if context.task_id == task.task_id]
            rows.append(
                {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "treatment_by_host": {},
                    "injected_bytes": {context.arm_id: context.injected_bytes for context in contexts},
                    "underperforming_baselines": [],
                    "findings": [],
                }
            )
        return rows
    rows = []
    for analysis in run.analysis.task_analyses:
        rows.append(
            {
                "task_id": analysis.task_id,
                "kind": run.task(analysis.task_id).kind,
                "treatment_by_host": {host: _outcome_summary(analysis.treatment_for(host)) for host in analysis.hosts},
                "injected_bytes": {outcome.arm_id: outcome.context.injected_bytes for outcome in analysis.outcomes},
                "underperforming_baselines": [
                    f"{outcome.arm_id}@{outcome.host}" for outcome in analysis.underperforming_baselines
                ],
                "findings": [
                    {
                        "arm_id": finding.arm_id,
                        "host": finding.host,
                        "failure_class": finding.failure_class,
                    }
                    for finding in analysis.findings
                ],
            }
        )
    return cast("list[dict[str, object]]", rows)


def _outcome_summary(outcome: ArmOutcome | None) -> str:
    if outcome is None or outcome.score is None:
        return "not recorded"
    score = outcome.score
    if not score.task_success:
        return "not recovered"
    return f"recovered at step {score.time_to_recover_state}"


def _failure_block(analysis: WorkContinuityAnalysis | None) -> dict[str, object]:
    if analysis is None:
        return {
            "available": False,
            "reason": "no recorded attempts were supplied, so no outcome could be classified",
            "findings": [],
            "findings_by_class": {},
            "underperforming_task_ids": [],
            "treatment_underperformance": [],
            "unrecorded_keys": [],
        }
    return {
        "available": True,
        "treatment_arm_id": TREATMENT_ARM_ID,
        "underperforming_task_ids": list(analysis.underperforming_task_ids),
        "treatment_underperformance": [
            {
                "task_id": entry.task_id,
                "host": entry.host,
                "beaten_by": list(entry.beaten_by),
                "treatment_failure_class": entry.treatment_failure_class,
                "requirement": entry.requirement,
                "recommendation": entry.recommendation,
            }
            for entry in analysis.underperformance
        ],
        "findings_by_class": analysis.findings_by_class,
        "unrecorded_keys": [list(key) for key in analysis.unrecorded_keys],
        "findings": [
            {
                "task_id": finding.task_id,
                "arm_id": finding.arm_id,
                "host": finding.host,
                "failure_class": finding.failure_class,
                "requirement": finding.requirement,
                "detail": finding.detail,
                "recommendation": finding.recommendation,
            }
            for finding in analysis.findings
        ],
    }


@dataclass
class _Recommendation:
    """One recommendation aggregated over every finding that asked for it."""

    failure_class: str
    requirement: str | None
    recommendation: str
    occurrences: int = 0
    tasks: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        return {
            "failure_class": self.failure_class,
            "requirement": self.requirement,
            "recommendation": self.recommendation,
            "occurrences": self.occurrences,
            "tasks": list(self.tasks),
            "arms": list(self.arms),
            "hosts": list(self.hosts),
        }


def _recommendations(analysis: WorkContinuityAnalysis | None) -> list[dict[str, object]]:
    """Collapse findings into one recommendation per failure class.

    ``no_recording`` is excluded: it asks for evidence rather than for a contract
    change, and mixing the two would make an incomplete run look like a finding.
    """

    if analysis is None:
        return []
    grouped: dict[str, _Recommendation] = {}
    for finding in analysis.findings:
        if finding.failure_class == NO_RECORDING:
            continue
        entry = grouped.setdefault(
            finding.failure_class,
            _Recommendation(
                failure_class=finding.failure_class,
                requirement=finding.requirement,
                recommendation=finding.recommendation,
            ),
        )
        entry.occurrences += 1
        _append_unique(entry.tasks, finding.task_id)
        _append_unique(entry.arms, finding.arm_id)
        _append_unique(entry.hosts, finding.host)
    return [grouped[key].as_json() for key in sorted(grouped)]


def _append_unique(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _boundaries(run: WorkContinuityRun) -> list[str]:
    boundaries = [
        (
            "The checked-in task set is authored fixture material for harness validation, not a sample of real "
            "user sessions, and its ground truth is declared rather than harvested."
        ),
        (
            "The harness runs no model: a recorded attempt is the only input that describes what a host did, and "
            "the recorder is responsible for mapping steps onto the task's declared fact ids."
        ),
        (
            "Injected bytes count the assembled continuation context only. A host's own system prompt, tool "
            "schemas, repository instructions, and workspace files are outside this measurement."
        ),
        (
            "Task success is scored against one declared next action per task, so a different but equally correct "
            "continuation is not credited."
        ),
    ]
    if run.analysis is None:
        boundaries.append("No recorded attempts were supplied, so this run reports assembly only.")
    else:
        boundaries.append(
            "Recorded hosts in this run are "
            + ", ".join(f"`{host}`" for host in run.hosts)
            + ". Names do not imply a verified host integration."
        )
    return boundaries


def _host_names(payload: dict[str, object]) -> list[str]:
    """Return the report's host names, or a placeholder when none recorded."""

    hosts = payload.get("hosts")
    if not isinstance(hosts, list) or any(not isinstance(host, str) for host in hosts):
        raise ReportError("report payload hosts must be an array of strings")
    names = [host for host in hosts if isinstance(host, str)]
    return names or ["none (assembly only)"]


def _mapping(value: object, label: str) -> dict[str, object]:
    """Return one report payload member as a JSON object mapping."""

    if not isinstance(value, dict):
        raise ReportError(f"report payload {label} must be an object")
    return cast("dict[str, object]", value)


def _entries(value: object, label: str) -> list[dict[str, object]]:
    """Return one report payload member as a list of JSON object mappings."""

    if not isinstance(value, list) or any(not isinstance(entry, dict) for entry in value):
        raise ReportError(f"report payload {label} must be an array of objects")
    return [cast("dict[str, object]", entry) for entry in value]


def _inline_counts(counts: object) -> str:
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(_mapping(counts, "findings_by_class").items()))
    return rendered or "none"
