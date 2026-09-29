"""Verify strict experiment configuration, prompt rendering, fingerprint, and metadata association."""

import json
from pathlib import Path

import pytest
import yaml

from agent.prompt_builder import (
    PromptBuilder,
    PromptTemplateError,
    build_implementation_phase_prompt,
    build_recovery_edit_gate_prompt,
    build_recovery_implementation_prompt,
    build_recovery_read_gate_prompt,
    build_verification_phase_prompt,
)
from benchmark.task import SWEbenchTask
from experiment.config import ConfigurationError, ExperimentConfig
from tracking.run_manager import RunManager


# Derive the project root from the test file location to avoid depending on the current directory when running pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_task() -> SWEbenchTask:
    """Build a task containing a dollar sign to ensure the Template does not expand the problem text a second time."""

    return SWEbenchTask(
        instance_id="example__project-789",
        repo="example/project",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix $HOME handling without reading a reference patch.",
    )


@pytest.mark.parametrize(
    ("name", "phase", "expected_frozen"),
    [("dev.yaml", "dev", False), ("evaluation.yaml", "evaluation", True)],
)
def test_repository_configs_have_expected_freeze_state(
    name, phase, expected_frozen
) -> None:
    """The Dev config stays adjustable, while the official Evaluation config must be frozen once development is complete."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / name)

    assert config.experiment.phase == phase
    assert config.experiment.configuration_frozen is expected_frozen
    # Prevent the config from accidentally falling back to an older model that cannot produce structured tool_use.
    assert config.model.name == "qwen3.5:9b"
    assert config.model.max_output_tokens == 8192
    assert config.model.max_output_tokens < config.model.context_length
    assert config.agent.max_turns == 30
    assert config.network.formal_solving is False
    # The Dev phase keeps a single worker for easier debugging; the official evaluation uses two workers to use CPU capacity while avoiding excessive contention on 20GB of WSL memory.
    expected_workers = 1 if phase == "dev" else 2
    assert config.evaluation.max_workers == expected_workers
    assert config.evaluation.cache_level == "env"
    assert len(config.fingerprint) == 64
    assert config.prompt_template_path.is_file()
    assert config.tasks_path.is_file()


def test_config_fingerprint_is_independent_of_yaml_key_order(tmp_path) -> None:
    """Only reordering YAML keys should not change the fingerprint of the normalized content."""

    source = PROJECT_ROOT / "configs" / "dev.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    reordered = {key: raw[key] for key in reversed(raw)}
    destination = PROJECT_ROOT / "configs" / "temporary-order-test.yaml"
    try:
        destination.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")
        assert ExperimentConfig.load(source).fingerprint == ExperimentConfig.load(destination).fingerprint
    finally:
        destination.unlink(missing_ok=True)


def test_config_rejects_network_during_formal_solving(tmp_path) -> None:
    """Enabling the network during the formal solving phase must be rejected to reduce the risk of solution leakage."""

    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dev.yaml").read_text(encoding="utf-8")
    )
    raw["network"]["formal_solving"] = True
    destination = PROJECT_ROOT / "configs" / "temporary-invalid-test.yaml"
    try:
        destination.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="solution leakage"):
            ExperimentConfig.load(destination)
    finally:
        destination.unlink(missing_ok=True)


def test_model_test_planning_turns_are_rejected(tmp_path) -> None:
    """Once the parent process takes over test planning, no turns may be reserved for an independent model planner."""

    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "dev_v2.yaml").read_text(encoding="utf-8")
    )
    raw["agent"]["test_planning_turns"] = 4
    destination = PROJECT_ROOT / "configs" / "temporary-no-planner-test.yaml"
    try:
        destination.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="parent-generated"):
            ExperimentConfig.load(destination)
    finally:
        destination.unlink(missing_ok=True)


def test_evaluation_config_is_frozen_for_formal_batch() -> None:
    """After Dev is complete, the repository's evaluation config must allow the official ten-task script to start."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")

    config.require_frozen()


def test_dev_v2_reserves_an_independent_verification_session() -> None:
    """The new architecture must keep verification turns and context guardrails while leaving the official baseline config unchanged."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml")

    assert config.experiment.phase == "dev"
    assert config.experiment.configuration_frozen is False
    assert config.agent.max_turns == 40
    assert config.agent.verification_turns == 10
    assert config.agent.test_planning_turns == 0
    assert config.agent.max_file_read_lines == 200
    assert config.agent.max_tool_output_chars == 12000
    assert config.agent.visible_test_sandbox is True
    assert config.agent.visible_test_timeout_seconds == 900
    assert config.agent.visible_regression_test_timeout_seconds == 300
    assert config.agent.task_timeout_seconds == 1800
    baseline = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
    assert baseline.agent.max_turns == 30
    assert baseline.agent.verification_turns == 0
    assert baseline.agent.visible_test_sandbox is False


def test_evaluation_v2_is_frozen_as_a_separate_ablation() -> None:
    """The architecture v2 ten-task config must be frozen, but must not override the experimental identity of the original official baseline."""

    baseline = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
    ablation = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation_v2.yaml")

    ablation.require_frozen()
    assert ablation.experiment.phase == "evaluation"
    assert ablation.experiment.name.endswith("evaluation-v2")
    assert ablation.fingerprint != baseline.fingerprint
    assert ablation.agent.max_turns == 40
    assert ablation.agent.verification_turns == 10
    assert ablation.agent.test_planning_turns == 0
    assert ablation.agent.visible_test_sandbox is True
    assert ablation.agent.visible_regression_test_timeout_seconds == 300
    assert ablation.agent.task_timeout_seconds == 1800


@pytest.mark.parametrize("task_count", [15, 20, 30])
def test_extended_evaluation_v2_configs_are_frozen_and_sized(task_count: int) -> None:
    """Extended evaluations must be frozen, and the config name, task list, and declared size must stay consistent."""

    config = ExperimentConfig.load(
        PROJECT_ROOT / "configs" / f"evaluation_v2_{task_count}_seed42.yaml"
    )
    task_ids = json.loads(config.tasks_path.read_text(encoding="utf-8"))

    config.require_frozen()
    assert config.experiment.name.endswith(f"{task_count}-seed42")
    assert len(task_ids) == task_count
    assert len(set(task_ids)) == task_count
    assert config.agent.task_timeout_seconds == 1800
    assert config.agent.visible_regression_test_timeout_seconds == 300


def test_seed43_tasks_are_frozen_and_disjoint_from_every_existing_set() -> None:
    """The new 30 tasks must be frozen and must not reuse any existing Dev/Evaluation instance in the repository."""

    config = ExperimentConfig.load(
        PROJECT_ROOT / "configs" / "evaluation_v2_30_seed43.yaml"
    )
    task_ids = json.loads(config.tasks_path.read_text(encoding="utf-8"))
    previous_ids: set[str] = set()
    for path in sorted((PROJECT_ROOT / "experiments").glob("*tasks*.json")):
        if path == config.tasks_path:
            continue
        previous_ids.update(json.loads(path.read_text(encoding="utf-8")))

    config.require_frozen()
    assert config.experiment.random_seed == 43
    assert config.experiment.name.endswith("30-seed43")
    assert len(task_ids) == len(set(task_ids)) == 30
    assert set(task_ids).isdisjoint(previous_ids)
    assert config.agent.task_timeout_seconds == 1800
    assert config.agent.visible_regression_test_timeout_seconds == 300


def test_phase_prompts_reanchor_task_and_enforce_delivery_boundaries() -> None:
    """Both independent sessions must carry the original task and separately emphasize delivering a patch and verifying the repair."""

    base = "instance_id: example__repo-1\nProblem: preserve semantics\n"
    implementation = build_implementation_phase_prompt(
        base,
        implementation_turns=30,
        verification_turns=10,
        max_file_read_lines=200,
        max_tool_output_chars=12000,
    )
    verification = build_verification_phase_prompt(
        base,
        verification_turns=10,
        max_file_read_lines=200,
        max_tool_output_chars=12000,
        candidate_patch="diff --git a/a.py b/a.py\n-old\n+new\n",
        scheduled_test_evidence="Exit code: 1\nFAILED expected value",
        focused_new_regression=True,
    )
    read_gate = build_recovery_read_gate_prompt(
        base,
        implementation_handoff="Likely defect in src/example.py.",
        source_context="File: src/example.py\n1: old_value = 1",
    )
    edit_gate = build_recovery_edit_gate_prompt(
        base,
        target_file="src/example.py",
    )
    recovery = build_recovery_implementation_prompt(
        base,
        recovery_turns=10,
        max_file_read_lines=200,
        max_tool_output_chars=12000,
        implementation_handoff=(
            "Previous termination: reason=max_turns; subtype=error_max_turns\n"
            "- Read: file_path=src/example.py"
        ),
    )

    assert base.strip() in implementation
    assert "non-empty candidate patch" in implementation
    assert "at most 200 source lines" in implementation
    assert ".agent-test-plan.json" in implementation
    assert "parent scheduler owns test execution" in implementation
    assert "Do not create `.agent-test-plan.json`" in implementation
    assert base.strip() in verification
    assert "Candidate patch collected relative" in verification
    assert "diff --git a/a.py b/a.py" in verification
    assert "FAILED expected value" in verification
    assert "Bash is disabled" in verification
    assert "Focused repair mode" in verification
    assert "Only Read and Edit are available" in verification
    assert "missing plan" in verification
    assert "hidden SWE-bench tests" in verification
    assert base.strip() in recovery
    assert "empty-patch recovery fallback" in recovery
    assert "at most 10 turns" in recovery
    assert "reason=max_turns" in recovery
    assert "file_path=src/example.py" in recovery
    assert "Bash is disabled" in recovery
    assert "failed sequence validation" in recovery
    assert "Modify existing product source" in recovery
    assert "do not run tests" in recovery
    assert base.strip() in read_gate
    assert "mandatory Read step" in read_gate
    assert "call Read exactly once" in read_gate
    assert "old_value = 1" in read_gate
    assert base.strip() in edit_gate
    assert "mandatory Edit step" in edit_gate
    assert "call Edit exactly once" in edit_gate
    assert "src/example.py" in edit_gate


def test_dev_config_cannot_be_used_as_formal_evaluation() -> None:
    """A Dev config that may still be tuned must not bypass the formal evaluation freeze boundary."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")

    with pytest.raises(ConfigurationError, match="not frozen"):
        config.require_frozen()


def test_prompt_builder_renders_only_safe_task_fields() -> None:
    """The prompt should contain safe task fields, fixed constraints, and a normalized trailing newline."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")
    prompt = PromptBuilder.from_file(config.prompt_template_path).build(make_task())

    assert "example__project-789" in prompt
    assert "example/project" in prompt
    assert "Fix $HOME handling" in prompt
    assert "Do not inspect a gold patch" in prompt
    assert prompt.endswith("\n")


def test_prompt_builder_requires_all_placeholders() -> None:
    """A template missing any required task placeholder must fail at construction time."""

    with pytest.raises(PromptTemplateError, match="missing placeholder"):
        PromptBuilder("Only ${instance_id}")


def test_run_metadata_contains_configuration_fingerprint(tmp_path) -> None:
    """Each run's metadata must associate the exact configuration fingerprint and network policy."""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")
    task = make_task()
    session = RunManager(tmp_path / "runs").start(
        task,
        phase=config.experiment.phase,
        agent=config.agent.framework,
        model=config.model.name,
        prompt=PromptBuilder.from_file(config.prompt_template_path).build(task),
        configuration=config.to_metadata(),
    )

    metadata = json.loads((session.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["configuration"]["config_fingerprint"] == config.fingerprint
    assert metadata["configuration"]["network_during_solving"] is False
