import json
from pathlib import Path

import pytest
import yaml

from agent.prompt_builder import PromptBuilder, PromptTemplateError
from benchmark.task import SWEbenchTask
from experiment.config import ConfigurationError, ExperimentConfig
from tracking.run_manager import RunManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_task() -> SWEbenchTask:
    return SWEbenchTask(
        instance_id="example__project-789",
        repo="example/project",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix $HOME handling without reading a reference patch.",
    )


@pytest.mark.parametrize(
    ("name", "phase"),
    [("dev.yaml", "dev"), ("evaluation.yaml", "evaluation")],
)
def test_repository_configs_are_valid_and_not_prematurely_frozen(name, phase) -> None:
    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / name)

    assert config.experiment.phase == phase
    assert config.experiment.configuration_frozen is False
    assert config.model.name == "qwen2.5-coder:7b"
    assert config.network.formal_solving is False
    assert config.evaluation.max_workers == 1
    assert len(config.fingerprint) == 64
    assert config.prompt_template_path.is_file()
    assert config.tasks_path.is_file()


def test_config_fingerprint_is_independent_of_yaml_key_order(tmp_path) -> None:
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


def test_evaluation_config_cannot_be_used_as_frozen_yet() -> None:
    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")

    with pytest.raises(ConfigurationError, match="not frozen"):
        config.require_frozen()


def test_prompt_builder_renders_only_safe_task_fields() -> None:
    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")
    prompt = PromptBuilder.from_file(config.prompt_template_path).build(make_task())

    assert "example__project-789" in prompt
    assert "example/project" in prompt
    assert "Fix $HOME handling" in prompt
    assert "Do not inspect a gold patch" in prompt
    assert prompt.endswith("\n")


def test_prompt_builder_requires_all_placeholders() -> None:
    with pytest.raises(PromptTemplateError, match="missing placeholder"):
        PromptBuilder("Only ${instance_id}")


def test_run_metadata_contains_configuration_fingerprint(tmp_path) -> None:
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
