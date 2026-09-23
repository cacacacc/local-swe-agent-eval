"""Strict, reproducible YAML configuration for local SWE-bench experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is missing or inconsistent."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    location: str,
    expected: set[str],
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    messages: list[str] = []
    if missing:
        messages.append(f"missing: {', '.join(sorted(missing))}")
    if unknown:
        messages.append(f"unknown: {', '.join(sorted(unknown))}")
    if messages:
        raise ConfigurationError(f"invalid keys in {location} ({'; '.join(messages)})")


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{location} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{location} must be true or false")
    return value


def _relative_path(value: Any, location: str) -> Path:
    path = Path(_string(value, location))
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(
            f"{location} must be a project-relative path without '..'"
        )
    return path


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    name: str
    phase: str
    random_seed: int
    configuration_frozen: bool


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    name: str
    split: str
    tasks_file: Path


@dataclass(frozen=True, slots=True)
class ModelSettings:
    runtime: str
    name: str
    context_length: int


@dataclass(frozen=True, slots=True)
class AgentSettings:
    framework: str
    timeout_seconds: int
    prompt_template: Path


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    environment_preparation: bool
    formal_solving: bool


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    max_workers: int
    cache_level: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    repository_cache: Path
    workspaces: Path
    runs: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A validated config plus an order-independent content fingerprint."""

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
        source_path = Path(path).resolve()
        try:
            raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigurationError(f"cannot read config {source_path}: {error}") from error

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

        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != 1:
            raise ConfigurationError(
                f"unsupported schema_version {schema_version}; expected 1"
            )

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

        dataset_raw = _mapping(raw["dataset"], "dataset")
        _exact_keys(dataset_raw, "dataset", {"name", "split", "tasks_file"})
        dataset = DatasetSettings(
            name=_string(dataset_raw["name"], "dataset.name"),
            split=_string(dataset_raw["split"], "dataset.split"),
            tasks_file=_relative_path(dataset_raw["tasks_file"], "dataset.tasks_file"),
        )

        model_raw = _mapping(raw["model"], "model")
        _exact_keys(model_raw, "model", {"runtime", "name", "context_length"})
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
        )

        agent_raw = _mapping(raw["agent"], "agent")
        _exact_keys(
            agent_raw,
            "agent",
            {"framework", "timeout_seconds", "prompt_template"},
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
            prompt_template=_relative_path(
                agent_raw["prompt_template"],
                "agent.prompt_template",
            ),
        )

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

        project_root = source_path.parent.parent
        prompt_path = project_root / agent.prompt_template
        tasks_path = project_root / dataset.tasks_file
        if not prompt_path.is_file():
            raise ConfigurationError(f"prompt template does not exist: {prompt_path}")
        if not tasks_path.is_file():
            raise ConfigurationError(f"tasks file does not exist: {tasks_path}")

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
        return self.project_root / self.agent.prompt_template

    @property
    def tasks_path(self) -> Path:
        return self.project_root / self.dataset.tasks_file

    def require_frozen(self) -> None:
        if not self.experiment.configuration_frozen:
            raise ConfigurationError(
                "configuration is not frozen; formal evaluation must not start"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return the experiment settings that must accompany every run."""

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
            "agent_framework": self.agent.framework,
            "timeout_seconds": self.agent.timeout_seconds,
            "network_during_solving": self.network.formal_solving,
            "evaluation_max_workers": self.evaluation.max_workers,
            "evaluation_cache_level": self.evaluation.cache_level,
        }
