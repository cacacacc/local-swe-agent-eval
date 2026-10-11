"""Protect cumulative handoff state across model edits and parent-owned test runs."""

from agent.handoff import HandoffLedger


PATCH_ONE = "diff --git a/widget.py b/widget.py\n--- a/widget.py\n+++ b/widget.py\n-old\n+new\n"
PATCH_TWO = "diff --git a/widget.py b/widget.py\n--- a/widget.py\n+++ b/widget.py\n-old\n+newer\n"


def test_handoff_accumulates_stages_and_invalidates_old_patch_tests() -> None:
    """A later edit must not inherit a successful test result from the preceding patch."""

    handoff = HandoffLedger()
    handoff.record_model("implementation", patch=PATCH_ONE, exit_code=0, agent_reported="Located widget")
    handoff.record_test(
        "scheduled_test_initial",
        patch=PATCH_ONE,
        plan_status="generated",
        comparisons=[
            {"label": "target", "comparison": "both_passed"},
            {"label": "regression", "comparison": "both_passed"},
        ],
    )
    first_sha = handoff.current["patch"]["sha256"]
    assert handoff.current["test"]["status"] == "passed"

    handoff.record_model("verification", patch=PATCH_TWO, exit_code=0)

    assert handoff.current["patch"]["sha256"] != first_sha
    assert handoff.current["test"]["status"] == "pending"
    assert handoff.current["test"]["comparisons"] == []
    assert handoff.history[1]["test"]["status"] == "passed"
    assert len(handoff.snapshot()["history"]) == 3


def test_handoff_unavailable_evidence_never_claims_pass_and_prompt_is_bounded() -> None:
    """Missing plans and uncomparable Docker results remain unknown, not successes."""

    handoff = HandoffLedger()
    handoff.record_model("implementation", patch=PATCH_ONE, exit_code=0, agent_reported="x" * 5000)
    handoff.record_test(
        "scheduled_test_initial",
        patch=PATCH_ONE,
        plan_status="missing",
        comparisons=[{"label": "target", "comparison": "comparison_unavailable"}],
    )

    prompt = handoff.render_for_prompt(maximum_chars=500)
    assert handoff.current["test"]["status"] == "unavailable"
    assert "Current patch test: unavailable" in prompt
    assert len(prompt) <= 500
    assert "x" * 5000 not in prompt


def test_handoff_prioritizes_new_regression_over_other_comparisons() -> None:
    """The first new failure survives alongside a passing adjacent test."""

    handoff = HandoffLedger()
    handoff.record_model("implementation", patch=PATCH_ONE, exit_code=0)
    handoff.record_test(
        "scheduled_test_initial",
        patch=PATCH_ONE,
        plan_status="generated",
        comparisons=[
            {"label": "target", "comparison": "both_passed"},
            {"label": "regression", "comparison": "new_regression", "first_new_failure": "test_other"},
        ],
    )

    assert handoff.current["test"]["status"] == "new_regression"
    assert "test_other" in handoff.render_for_prompt()
