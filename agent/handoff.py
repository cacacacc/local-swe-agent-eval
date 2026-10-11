"""Maintain an auditable, bounded handoff across model and parent-test stages."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


class HandoffLedger:
    """Keep complete stage history while rendering only current, relevant evidence."""

    def __init__(self) -> None:
        """Start with no patch or test evidence for this task."""

        self.history: list[dict[str, Any]] = []
        self.current: dict[str, Any] = {
            "patch": {"sha256": None, "status": "absent", "files": []},
            "test": {"status": "not_run", "patch_sha256": None, "comparisons": []},
            "last_stage": None,
            "next_action": "Implement a source change.",
        }

    def _update_patch(self, patch: str) -> None:
        """Invalidate current test evidence whenever the actual candidate patch changes."""

        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch.strip() else None
        previous = self.current["patch"]["sha256"]
        files: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("+++ b/"):
                continue
            name = line[6:]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not name or len(name) > 240:
                continue
            if name not in files:
                files.append(name)
        self.current["patch"] = {
            "sha256": digest,
            "status": "present" if digest else "absent",
            "files": files[:12],
        }
        if digest != previous:
            # Historical comparisons remain in history, but cannot certify a newly edited patch.
            self.current["test"] = {
                "status": "pending" if digest else "not_run",
                "patch_sha256": digest,
                "comparisons": [],
            }

    def record_model(
        self,
        stage: str,
        *,
        patch: str,
        exit_code: int,
        agent_reported: str = "",
    ) -> None:
        """Record a model stage without treating its own diagnosis as verified fact."""

        self._update_patch(patch)
        note = agent_reported.strip()[:1200]
        event = {
            "sequence": len(self.history) + 1,
            "stage": stage,
            "kind": "model",
            "patch_sha256": self.current["patch"]["sha256"],
            "exit_code": exit_code,
            "agent_reported_unverified": note,
        }
        self.history.append(event)
        self.current["last_stage"] = stage
        if self.current["patch"]["status"] == "absent":
            self.current["next_action"] = "Create an existing-source patch."
        elif self.current["test"]["status"] == "pending":
            self.current["next_action"] = "Run parent-owned tests for the current patch."

    def record_test(
        self,
        stage: str,
        *,
        patch: str,
        plan_status: str,
        comparisons: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record only structured baseline/candidate comparisons for this exact patch."""

        self._update_patch(patch)
        safe_comparisons = [
            {
                "label": str(item.get("label", ""))[:80],
                "comparison": str(item.get("comparison", "comparison_unavailable")),
                "first_new_failure": str(item.get("first_new_failure", ""))[:240],
            }
            for item in comparisons
        ]
        kinds = {item["comparison"] for item in safe_comparisons}
        if "new_regression" in kinds:
            status = "new_regression"
            next_action = "Repair the first new regression, then rerun parent-owned tests."
        elif (
            plan_status not in {"accepted", "generated"}
            or len(safe_comparisons) != 2
            or "comparison_unavailable" in kinds
        ):
            status = "unavailable"
            next_action = "Test evidence is unavailable; preserve or inspect the current patch."
        elif "baseline_failure_persists" in kinds:
            status = "baseline_failure_persists"
            next_action = "Review persistent baseline failures without assuming a new regression."
        else:
            status = "passed"
            next_action = "Review the current patch and finish."
        test_state = {
            "status": status,
            "patch_sha256": self.current["patch"]["sha256"],
            "plan_status": plan_status,
            "comparisons": safe_comparisons,
        }
        self.current["test"] = test_state
        self.current["last_stage"] = stage
        self.current["next_action"] = next_action
        self.history.append({
            "sequence": len(self.history) + 1,
            "stage": stage,
            "kind": "parent_test",
            "patch_sha256": self.current["patch"]["sha256"],
            "test": dict(test_state),
        })

    def snapshot(self) -> dict[str, Any]:
        """Return the complete JSON-serializable history and current materialized state."""

        return {"schema_version": 1, "history": self.history, "current": self.current}

    def render_for_prompt(self, *, maximum_chars: int = 5000) -> str:
        """Prioritize current patch and test facts over older stage notes in a bounded prompt."""

        if maximum_chars < 300:
            raise ValueError("handoff prompt budget must be at least 300 characters")
        patch = self.current["patch"]
        test = self.current["test"]
        lines = [
            "Parent-maintained cumulative handoff (model notes are unverified):",
            f"Current patch: {patch['status']}; sha256={patch['sha256'] or 'none'}; files={', '.join(patch['files']) or 'none'}",
            f"Current patch test: {test['status']}; plan={test.get('plan_status', 'not_run')}",
        ]
        for item in test["comparisons"]:
            lines.append(
                f"- {item['label']}: {item['comparison']}"
                + (f"; first new failure={item['first_new_failure']}" if item["first_new_failure"] else "")
            )
        lines.append(f"Next action: {self.current['next_action']}")
        rendered = "\n".join(lines)
        if len(rendered) > maximum_chars:
            # Drop individual comparisons before core patch identity and test classification.
            lines = lines[:3] + [lines[-1]]
            rendered = "\n".join(lines)
        for event in reversed(self.history[-3:]):
            addition = f"\nPrior stage {event['sequence']}: {event['stage']} ({event['kind']})"
            note = event.get("agent_reported_unverified", "")
            if note:
                addition += f"; agent-reported (unverified): {note.replace(chr(10), ' ')[:400]}"
            if len(rendered) + len(addition) > maximum_chars:
                continue
            rendered += addition
        return rendered[:maximum_chars]
