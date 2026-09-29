"""Parse and validate the strict YAML configuration for local SWE-bench experiments.

This module uses a "reject unknown fields" schema policy so that configuration typos
are not silently ignored; it also computes a SHA-256 fingerprint over the normalized
configuration so every run can be traced to the exact parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when the experiment configuration has missing fields, wrong types, or violates experiment constraints."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Require the configuration node to be a mapping and preserve its location in error messages."""

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    location: str,
    expected: set[str],
) -> None:
    """Require the mapping's key set to match the schema exactly.

    Reports both missing and unknown keys, so YAML typos cannot silently pass through
    as default values.
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
    """Validate required and optional keys while still rejecting any unknown configuration.

    This helper is used only for backward compatibility with already-frozen v1
    experiment configurations. New Agent architecture parameters may appear only in
    later Dev configurations, but typos must still fail immediately rather than
    silently fall back.
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
    """Read a non-empty string and strip leading and trailing whitespace."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    """Read a bounded-from-below integer; explicitly exclude bool, which is an int subclass in Python."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, location: str) -> bool:
    """Require native YAML booleans rather than easily misread strings."""

    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be true or false")
    return value


def _relative_path(value: Any, location: str) -> Path:
    """Read a project-relative path, forbidding absolute paths and ``..`` path escapes."""

    path = Path(_string(value, location))
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(
            f"{location} must be a project-relative path without '..'"
        )
    return path


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Experiment identity, phase, random seed, and whether the configuration is frozen."""

    name: str
    phase: str
    random_seed: int
    configuration_frozen: bool


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Dataset name, split, and the location of the pinned task list."""

    name: str
    split: str
    tasks_file: Path


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Local model runtime, model label, context window, and per-output limit."""

    runtime: str
    name: str
    context_length: int
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Agent framework, per-task budget, context guardrails, and the pinned Prompt template."""

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
    visible_regression_test_timeout_seconds: int
    task_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """Control network access separately for the environment-preparation and formal-solving phases."""

    environment_preparation: bool
    formal_solving: bool


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Concurrency and Docker image cache level for the SWE-bench harness."""

    max_workers: int
    cache_level: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Project-relative paths for the repository cache, task worktrees, and run artifacts."""

    repository_cache: Path
    workspaces: Path
    runs: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The complete validated configuration plus a content fingerprint that is independent of YAML key order."""

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
        """Load the configuration from a YAML file and complete structural, semantic, and related-file validation."""

        source_path = Path(path).resolve()
        try:
            raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(f"cannot read config {source_path}: {error}") from error

        # Lock the schema at the first level, so future parameter additions are not silently ignored by older code.
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

        # schema_version reserves an explicit migration boundary for future incompatible format upgrades.
        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != 1:
            raise ConfigurationError(
                f"unsupported schema_version {schema_version}; expected 1"
            )

        # The experiment section decides whether this is a tunable dev run or a frozen evaluation.
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

        # tasks_file holds the pre-selected instance IDs, preventing tasks from being swapped after results are seen.
        dataset_raw = _mapping(raw["dataset"], "dataset")
        _exact_keys(dataset_raw, "dataset", {"name", "split", "tasks_file"})
        dataset = DatasetSettings(
            name=_string(dataset_raw["name"], "dataset.name"),
            split=_string(dataset_raw["split"], "dataset.split"),
            tasks_file=_relative_path(dataset_raw["tasks_file"], "dataset.tasks_file"),
        )

        # The current research question is limited to local Ollama inference; cloud providers are not accepted.
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
        # The output budget must be significantly smaller than the total window; otherwise tool
        # results crowd out the input space and trigger auto-compact repeatedly within a few turns,
        # causing Claude Code to terminate the run with rapid_refill_breaker.
        if model.max_output_tokens >= model.context_length:
            raise ConfigurationError(
                "model.max_output_tokens must be smaller than model.context_length"
            )

        # Pin the Agent framework and prompt file so that only the issue content varies across tasks.
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
                "visible_regression_test_timeout_seconds",
                "task_timeout_seconds",
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
            # Old frozen configurations lack these fields, and the defaults preserve the original
            # single-session behavior; new v2 configurations explicitly retain the test-planning and
            # verification budgets and enable the tool-output guardrails.
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
            # When old configurations do not declare it, reuse the original unified test timeout so the
            # run semantics behind historical fingerprints stay unchanged; new v2 configurations explicitly
            # shorten the wait ceiling for adjacent regression tests.
            visible_regression_test_timeout_seconds=_integer(
                agent_raw.get(
                    "visible_regression_test_timeout_seconds",
                    agent_raw.get("visible_test_timeout_seconds", 0),
                ),
                "agent.visible_regression_test_timeout_seconds",
                minimum=0,
            ),
            # 0 is only a compatibility-off value for old configurations; new architecture configurations must give an explicit total wall-clock budget.
            task_timeout_seconds=_integer(
                agent_raw.get("task_timeout_seconds", 0),
                "agent.task_timeout_seconds",
                minimum=0,
            ),
        )
        reserved_turns = agent.verification_turns
        if reserved_turns >= agent.max_turns:
            raise ConfigurationError(
                "agent verification turns must leave at least one implementation turn"
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
        if (
            agent.visible_test_sandbox
            and agent.visible_regression_test_timeout_seconds <= 0
        ):
            raise ConfigurationError(
                "visible test sandbox requires positive "
                "visible_regression_test_timeout_seconds"
            )
        if agent.test_planning_turns:
            raise ConfigurationError(
                "agent.test_planning_turns must be 0; test plans are parent-generated"
            )
        if agent.verification_turns and not agent.visible_test_sandbox:
            raise ConfigurationError(
                "verification requires visible_test_sandbox"
            )

        # Environment preparation may use the network to download dependencies; formal solving must stay offline to reduce answer-leakage risk.
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

        # Concurrency and image caching affect resource consumption and runtime, so they are included in the fingerprint as well.
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

        # All writable directories must live under project-relative paths for easy migration and cleanup.
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

        # The config file lives under ``configs/``; its parent's parent is the project root.
        project_root = source_path.parent.parent
        prompt_path = project_root / agent.prompt_template
        tasks_path = project_root / dataset.tasks_file
        if not prompt_path.is_file():
            raise ConfigurationError(f"prompt template does not exist: {prompt_path}")
        if not tasks_path.is_file():
            raise ConfigurationError(f"tasks file does not exist: {tasks_path}")

        # Sort keys and strip insignificant whitespace so YAML key order does not affect the fingerprint.
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
        """Return the absolute path of the Prompt template resolved against the project root."""

        return self.project_root / self.agent.prompt_template

    @property
    def tasks_path(self) -> Path:
        """Return the absolute path of the pinned task list resolved against the project root."""

        return self.project_root / self.dataset.tasks_file

    def require_frozen(self) -> None:
        """Prevent unfrozen configurations from entering the formal evaluation phase."""

        if not self.experiment.configuration_frozen:
            raise ConfigurationError(
                "configuration is not frozen; formal evaluation must not start"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return the experiment configuration summary that must be saved with every run's artifacts."""

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
            ),
            "max_file_read_lines": self.agent.max_file_read_lines,
            "max_tool_output_chars": self.agent.max_tool_output_chars,
            "visible_test_sandbox": self.agent.visible_test_sandbox,
            "visible_test_timeout_seconds": self.agent.visible_test_timeout_seconds,
            "visible_regression_test_timeout_seconds": (
                self.agent.visible_regression_test_timeout_seconds
            ),
            "task_timeout_seconds": self.agent.task_timeout_seconds,
            "network_during_solving": self.network.formal_solving,
            "evaluation_max_workers": self.evaluation.max_workers,
            "evaluation_cache_level": self.evaluation.cache_level,
        }
