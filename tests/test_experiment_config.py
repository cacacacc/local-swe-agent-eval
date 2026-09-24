"""验证严格实验配置、Prompt 渲染、fingerprint 与 metadata 关联。"""

import json
from pathlib import Path

import pytest
import yaml

from agent.prompt_builder import (
    PromptBuilder,
    PromptTemplateError,
    build_implementation_phase_prompt,
    build_verification_phase_prompt,
)
from benchmark.task import SWEbenchTask
from experiment.config import ConfigurationError, ExperimentConfig
from tracking.run_manager import RunManager


# 从测试文件位置推导项目根目录，避免依赖执行 pytest 时的当前目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_task() -> SWEbenchTask:
    """构造包含美元符号的任务，确保 Template 不会二次展开题目文本。"""

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
    """Dev 配置保持可调，而正式 Evaluation 配置必须在开发完成后冻结。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / name)

    assert config.experiment.phase == phase
    assert config.experiment.configuration_frozen is expected_frozen
    # 防止配置意外退回无法产生结构化 tool_use 的旧模型。
    assert config.model.name == "qwen3.5:9b"
    assert config.model.max_output_tokens == 8192
    assert config.model.max_output_tokens < config.model.context_length
    assert config.agent.max_turns == 30
    assert config.network.formal_solving is False
    # Dev 阶段保持单 worker 便于排错；正式评测使用两个 worker，以利用 CPU 且避免 20GB WSL 内存发生过度争用。
    expected_workers = 1 if phase == "dev" else 2
    assert config.evaluation.max_workers == expected_workers
    assert config.evaluation.cache_level == "env"
    assert len(config.fingerprint) == 64
    assert config.prompt_template_path.is_file()
    assert config.tasks_path.is_file()


def test_config_fingerprint_is_independent_of_yaml_key_order(tmp_path) -> None:
    """仅调整 YAML 键顺序不应改变规范化内容的 fingerprint。"""

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
    """正式求解阶段启用网络必须被拒绝，以降低 solution leakage 风险。"""

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


def test_evaluation_config_is_frozen_for_formal_batch() -> None:
    """完成 Dev 后，仓库中的 evaluation 配置必须允许正式十题脚本启动。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")

    config.require_frozen()


def test_dev_v2_reserves_an_independent_verification_session() -> None:
    """新架构必须保留验证 turns 和上下文护栏，同时不得修改正式基线配置。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml")

    assert config.experiment.phase == "dev"
    assert config.experiment.configuration_frozen is False
    assert config.agent.max_turns == 40
    assert config.agent.verification_turns == 10
    assert config.agent.max_file_read_lines == 200
    assert config.agent.max_tool_output_chars == 12000
    assert config.agent.visible_test_sandbox is True
    assert config.agent.visible_test_timeout_seconds == 900
    baseline = ExperimentConfig.load(PROJECT_ROOT / "configs" / "evaluation.yaml")
    assert baseline.agent.max_turns == 30
    assert baseline.agent.verification_turns == 0
    assert baseline.agent.visible_test_sandbox is False


def test_phase_prompts_reanchor_task_and_enforce_delivery_boundaries() -> None:
    """两个独立会话都必须携带原任务，且分别强调交付补丁和验证修复。"""

    base = "instance_id: example__repo-1\nProblem: preserve semantics\n"
    implementation = build_implementation_phase_prompt(
        base,
        implementation_turns=30,
        verification_turns=10,
        max_file_read_lines=200,
        max_tool_output_chars=12000,
        visible_test_command="python sandbox.py --",
    )
    verification = build_verification_phase_prompt(
        base,
        verification_turns=10,
        max_file_read_lines=200,
        max_tool_output_chars=12000,
        visible_test_command="python sandbox.py --",
        candidate_patch="diff --git a/a.py b/a.py\n-old\n+new\n",
    )

    assert base.strip() in implementation
    assert "non-empty candidate patch" in implementation
    assert "at most 200 source lines" in implementation
    assert "python sandbox.py --" in implementation
    assert base.strip() in verification
    assert "Candidate patch collected relative" in verification
    assert "diff --git a/a.py b/a.py" in verification
    assert "first tool action must run" in verification
    assert "hidden SWE-bench tests" in verification
    assert "network-disabled visible-test" in verification


def test_dev_config_cannot_be_used_as_formal_evaluation() -> None:
    """可继续调参的 Dev 配置仍不得绕过正式评测冻结边界。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")

    with pytest.raises(ConfigurationError, match="not frozen"):
        config.require_frozen()


def test_prompt_builder_renders_only_safe_task_fields() -> None:
    """Prompt 应包含安全任务字段、固定约束和规范化结尾换行。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev.yaml")
    prompt = PromptBuilder.from_file(config.prompt_template_path).build(make_task())

    assert "example__project-789" in prompt
    assert "example/project" in prompt
    assert "Fix $HOME handling" in prompt
    assert "Do not inspect a gold patch" in prompt
    assert prompt.endswith("\n")


def test_prompt_builder_requires_all_placeholders() -> None:
    """缺少任一必要任务占位符的模板必须在构造时失败。"""

    with pytest.raises(PromptTemplateError, match="missing placeholder"):
        PromptBuilder("Only ${instance_id}")


def test_run_metadata_contains_configuration_fingerprint(tmp_path) -> None:
    """每次运行的 metadata 必须关联准确配置 fingerprint 和网络策略。"""

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
