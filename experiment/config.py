"""解析并验证本地 SWE-bench 实验的严格 YAML 配置。

本模块采用“拒绝未知字段”的 schema 策略，防止配置拼写错误被静默忽略；同时
对规范化后的配置计算 SHA-256 fingerprint，使每次运行都能追溯到准确参数集合。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """当实验配置缺失字段、类型错误或违反实验约束时抛出。"""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    """要求配置节点是映射，并在错误信息中保留节点位置。"""

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    location: str,
    expected: set[str],
) -> None:
    """要求映射的键集合与 schema 完全一致。

    同时报告缺失键和未知键，使 YAML 拼写错误不会以默认值悄悄通过。
    """

    missing = expected - set(value)
    unknown = set(value) - expected
    messages: list[str] = []
    if missing:
        messages.append(f"missing: {', '.join(sorted(missing))}")
    if unknown:
        messages.append(f"unknown: {', '.join(sorted(unknown))}")
    if messages:
        raise ConfigurationError(f"invalid keys in {location} ({'; '.join(messages)})")


def _required_and_optional_keys(
    value: Mapping[str, Any],
    location: str,
    *,
    required: set[str],
    optional: set[str],
) -> None:
    """校验必填键与可选键，同时继续拒绝任何未知配置。

    该辅助函数只用于向后兼容已经冻结的 v1 实验配置。新增的 Agent 架构参数
    可以仅出现在后续 Dev 配置中，但拼写错误仍然必须立即失败，不能静默回退。
    """

    missing = required - set(value)
    unknown = set(value) - required - optional
    messages: list[str] = []
    if missing:
        messages.append(f"missing: {', '.join(sorted(missing))}")
    if unknown:
        messages.append(f"unknown: {', '.join(sorted(unknown))}")
    if messages:
        raise ConfigurationError(f"invalid keys in {location} ({'; '.join(messages)})")


def _string(value: Any, location: str) -> str:
    """读取非空字符串，并移除首尾空白。"""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    """读取有下界的整数；显式排除 Python 中属于 int 子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, location: str) -> bool:
    """要求使用 YAML 原生布尔值，而不是容易误解的字符串。"""

    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be true or false")
    return value


def _relative_path(value: Any, location: str) -> Path:
    """读取项目内相对路径，禁止绝对路径和 ``..`` 路径逃逸。"""

    path = Path(_string(value, location))
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(
            f"{location} must be a project-relative path without '..'"
        )
    return path


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """实验身份、阶段、随机种子以及配置是否冻结。"""

    name: str
    phase: str
    random_seed: int
    configuration_frozen: bool


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """数据集名称、split 和固定题目清单的位置。"""

    name: str
    split: str
    tasks_file: Path


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """本地模型运行时、模型标签、上下文窗口和单次输出上限。"""

    runtime: str
    name: str
    context_length: int
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Agent 框架、单题预算、上下文护栏和固定 Prompt 模板。"""

    framework: str
    timeout_seconds: int
    max_turns: int
    prompt_template: Path
    verification_turns: int
    test_planning_turns: int
    max_file_read_lines: int
    max_tool_output_chars: int
    visible_test_sandbox: bool
    visible_test_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """分别控制环境准备阶段与正式求解阶段的网络权限。"""

    environment_preparation: bool
    formal_solving: bool


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """SWE-bench harness 的并发数和 Docker image 缓存级别。"""

    max_workers: int
    cache_level: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """仓库缓存、任务 worktree 和运行产物的项目内相对路径。"""

    repository_cache: Path
    workspaces: Path
    runs: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """完整的已验证配置，以及与 YAML 键顺序无关的内容 fingerprint。"""

    schema_version: int
    experiment: ExperimentSettings
    dataset: DatasetSettings
    model: ModelSettings
    agent: AgentSettings
    network: NetworkPolicy
    evaluation: EvaluationSettings
    storage: StorageSettings
    source_path: Path
    project_root: Path
    fingerprint: str

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        """从 YAML 文件加载配置，完成结构、语义和关联文件验证。"""

        source_path = Path(path).resolve()
        try:
            raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(f"cannot read config {source_path}: {error}") from error

        # 第一层先锁定 schema，避免未来增加参数时旧代码静默忽略它们。
        raw = _mapping(raw_value, "config")
        top_level_keys = {
            "schema_version",
            "experiment",
            "dataset",
            "model",
            "agent",
            "network",
            "evaluation",
            "storage",
        }
        _exact_keys(raw, "config", top_level_keys)

        # schema_version 为未来不兼容格式升级保留明确的迁移边界。
        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != 1:
            raise ConfigurationError(
                f"unsupported schema_version {schema_version}; expected 1"
            )

        # experiment 段决定当前是允许调参的 dev，还是冻结后的 evaluation。
        experiment_raw = _mapping(raw["experiment"], "experiment")
        _exact_keys(
            experiment_raw,
            "experiment",
            {"name", "phase", "random_seed", "configuration_frozen"},
        )
        phase = _string(experiment_raw["phase"], "experiment.phase")
        if phase not in {"dev", "evaluation"}:
            raise ConfigurationError("experiment.phase must be 'dev' or 'evaluation'")
        experiment = ExperimentSettings(
            name=_string(experiment_raw["name"], "experiment.name"),
            phase=phase,
            random_seed=_integer(
                experiment_raw["random_seed"],
                "experiment.random_seed",
            ),
            configuration_frozen=_boolean(
                experiment_raw["configuration_frozen"],
                "experiment.configuration_frozen",
            ),
        )

        # tasks_file 保存预先选定的 instance ID，避免见到结果后替换题目。
        dataset_raw = _mapping(raw["dataset"], "dataset")
        _exact_keys(dataset_raw, "dataset", {"name", "split", "tasks_file"})
        dataset = DatasetSettings(
            name=_string(dataset_raw["name"], "dataset.name"),
            split=_string(dataset_raw["split"], "dataset.split"),
            tasks_file=_relative_path(dataset_raw["tasks_file"], "dataset.tasks_file"),
        )

        # 当前研究问题限定为 Ollama 本地推理，不接受云端 provider。
        model_raw = _mapping(raw["model"], "model")
        _exact_keys(
            model_raw,
            "model",
            {"runtime", "name", "context_length", "max_output_tokens"},
        )
        runtime = _string(model_raw["runtime"], "model.runtime")
        if runtime != "ollama":
            raise ConfigurationError("model.runtime must be 'ollama'")
        model = ModelSettings(
            runtime=runtime,
            name=_string(model_raw["name"], "model.name"),
            context_length=_integer(
                model_raw["context_length"],
                "model.context_length",
                minimum=4096,
            ),
            max_output_tokens=_integer(
                model_raw["max_output_tokens"],
                "model.max_output_tokens",
                minimum=1,
            ),
        )
        # 输出预算必须显著小于总窗口，否则工具结果会挤占输入空间并导致 auto-compact
        # 在少数轮次内反复触发，Claude Code 会以 rapid_refill_breaker 终止运行。
        if model.max_output_tokens >= model.context_length:
            raise ConfigurationError(
                "model.max_output_tokens must be smaller than model.context_length"
            )

        # 固定 Agent 框架和 prompt 文件，确保跨任务只改变 issue 内容。
        agent_raw = _mapping(raw["agent"], "agent")
        _required_and_optional_keys(
            agent_raw,
            "agent",
            required={
                "framework",
                "timeout_seconds",
                "max_turns",
                "prompt_template",
            },
            optional={
                "verification_turns",
                "test_planning_turns",
                "max_file_read_lines",
                "max_tool_output_chars",
                "visible_test_sandbox",
                "visible_test_timeout_seconds",
            },
        )
        framework = _string(agent_raw["framework"], "agent.framework")
        if framework != "claude-code":
            raise ConfigurationError("agent.framework must be 'claude-code'")
        agent = AgentSettings(
            framework=framework,
            timeout_seconds=_integer(
                agent_raw["timeout_seconds"],
                "agent.timeout_seconds",
                minimum=1,
            ),
            max_turns=_integer(
                agent_raw["max_turns"],
                "agent.max_turns",
                minimum=1,
            ),
            prompt_template=_relative_path(
                agent_raw["prompt_template"],
                "agent.prompt_template",
            ),
            # 旧的冻结配置没有这些字段，默认值保持原来的单会话行为；新的 v2
            # 配置显式保留测试规划与验证预算，并启用工具输出护栏。
            verification_turns=_integer(
                agent_raw.get("verification_turns", 0),
                "agent.verification_turns",
                minimum=0,
            ),
            test_planning_turns=_integer(
                agent_raw.get("test_planning_turns", 0),
                "agent.test_planning_turns",
                minimum=0,
            ),
            max_file_read_lines=_integer(
                agent_raw.get("max_file_read_lines", 0),
                "agent.max_file_read_lines",
                minimum=0,
            ),
            max_tool_output_chars=_integer(
                agent_raw.get("max_tool_output_chars", 0),
                "agent.max_tool_output_chars",
                minimum=0,
            ),
            visible_test_sandbox=_boolean(
                agent_raw.get("visible_test_sandbox", False),
                "agent.visible_test_sandbox",
            ),
            visible_test_timeout_seconds=_integer(
                agent_raw.get("visible_test_timeout_seconds", 0),
                "agent.visible_test_timeout_seconds",
                minimum=0,
            ),
        )
        reserved_turns = agent.verification_turns + agent.test_planning_turns
        if reserved_turns >= agent.max_turns:
            raise ConfigurationError(
                "agent verification and test planning turns must leave at least one "
                "implementation turn"
            )
        if agent.verification_turns and (
            agent.max_file_read_lines <= 0 or agent.max_tool_output_chars <= 0
        ):
            raise ConfigurationError(
                "phased agent requires positive max_file_read_lines and "
                "max_tool_output_chars"
            )
        if agent.visible_test_sandbox and agent.visible_test_timeout_seconds <= 0:
            raise ConfigurationError(
                "visible test sandbox requires positive visible_test_timeout_seconds"
            )
        if agent.test_planning_turns and not agent.visible_test_sandbox:
            raise ConfigurationError(
                "test planning turns require the visible test sandbox"
            )
        if agent.verification_turns and (
            not agent.visible_test_sandbox or agent.test_planning_turns <= 0
        ):
            raise ConfigurationError(
                "verification requires visible_test_sandbox and positive "
                "test_planning_turns"
            )

        # 环境准备可以联网下载依赖；正式求解必须离线以降低答案泄漏风险。
        network_raw = _mapping(raw["network"], "network")
        _exact_keys(
            network_raw,
            "network",
            {"environment_preparation", "formal_solving"},
        )
        network = NetworkPolicy(
            environment_preparation=_boolean(
                network_raw["environment_preparation"],
                "network.environment_preparation",
            ),
            formal_solving=_boolean(
                network_raw["formal_solving"],
                "network.formal_solving",
            ),
        )
        if network.formal_solving:
            raise ConfigurationError(
                "network.formal_solving must be false to prevent solution leakage"
            )

        # 并发和镜像缓存会影响资源消耗与运行时间，因此也纳入 fingerprint。
        evaluation_raw = _mapping(raw["evaluation"], "evaluation")
        _exact_keys(evaluation_raw, "evaluation", {"max_workers", "cache_level"})
        cache_level = _string(evaluation_raw["cache_level"], "evaluation.cache_level")
        if cache_level not in {"none", "base", "env", "instance"}:
            raise ConfigurationError(
                "evaluation.cache_level must be one of: none, base, env, instance"
            )
        evaluation = EvaluationSettings(
            max_workers=_integer(
                evaluation_raw["max_workers"],
                "evaluation.max_workers",
                minimum=1,
            ),
            cache_level=cache_level,
        )

        # 所有可写目录必须位于项目相对路径下，方便迁移与清理。
        storage_raw = _mapping(raw["storage"], "storage")
        _exact_keys(
            storage_raw,
            "storage",
            {"repository_cache", "workspaces", "runs"},
        )
        storage = StorageSettings(
            repository_cache=_relative_path(
                storage_raw["repository_cache"],
                "storage.repository_cache",
            ),
            workspaces=_relative_path(storage_raw["workspaces"], "storage.workspaces"),
            runs=_relative_path(storage_raw["runs"], "storage.runs"),
        )

        # 配置文件位于 ``configs/``，其父目录的父目录即项目根目录。
        project_root = source_path.parent.parent
        prompt_path = project_root / agent.prompt_template
        tasks_path = project_root / dataset.tasks_file
        if not prompt_path.is_file():
            raise ConfigurationError(f"prompt template does not exist: {prompt_path}")
        if not tasks_path.is_file():
            raise ConfigurationError(f"tasks file does not exist: {tasks_path}")

        # 对键排序并移除无意义空白，使 YAML 键顺序不影响 fingerprint。
        canonical = json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            schema_version=schema_version,
            experiment=experiment,
            dataset=dataset,
            model=model,
            agent=agent,
            network=network,
            evaluation=evaluation,
            storage=storage,
            source_path=source_path,
            project_root=project_root,
            fingerprint=fingerprint,
        )

    @property
    def prompt_template_path(self) -> Path:
        """返回解析到项目根目录下的 Prompt 模板绝对路径。"""

        return self.project_root / self.agent.prompt_template

    @property
    def tasks_path(self) -> Path:
        """返回解析到项目根目录下的固定任务清单绝对路径。"""

        return self.project_root / self.dataset.tasks_file

    def require_frozen(self) -> None:
        """阻止未冻结配置进入正式评测阶段。"""

        if not self.experiment.configuration_frozen:
            raise ConfigurationError(
                "configuration is not frozen; formal evaluation must not start"
            )

    def to_metadata(self) -> dict[str, Any]:
        """返回每次运行必须随产物保存的实验配置摘要。"""

        return {
            "config_fingerprint": self.fingerprint,
            "config_schema_version": self.schema_version,
            "experiment_name": self.experiment.name,
            "experiment_phase": self.experiment.phase,
            "configuration_frozen": self.experiment.configuration_frozen,
            "random_seed": self.experiment.random_seed,
            "dataset": self.dataset.name,
            "dataset_split": self.dataset.split,
            "model_runtime": self.model.runtime,
            "model_name": self.model.name,
            "context_length": self.model.context_length,
            "max_output_tokens": self.model.max_output_tokens,
            "agent_framework": self.agent.framework,
            "timeout_seconds": self.agent.timeout_seconds,
            "max_turns": self.agent.max_turns,
            "verification_turns": self.agent.verification_turns,
            "test_planning_turns": self.agent.test_planning_turns,
            "implementation_turns": (
                self.agent.max_turns
                - self.agent.verification_turns
                - self.agent.test_planning_turns
            ),
            "max_file_read_lines": self.agent.max_file_read_lines,
            "max_tool_output_chars": self.agent.max_tool_output_chars,
            "visible_test_sandbox": self.agent.visible_test_sandbox,
            "visible_test_timeout_seconds": self.agent.visible_test_timeout_seconds,
            "network_during_solving": self.network.formal_solving,
            "evaluation_max_workers": self.evaluation.max_workers,
            "evaluation_cache_level": self.evaluation.cache_level,
        }
