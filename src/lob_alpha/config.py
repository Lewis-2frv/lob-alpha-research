"""Typed research configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a research configuration violates a frozen invariant."""


def _strictly_increasing(values: tuple[int, ...], name: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ConfigurationError(f"{name} must contain positive values")
    if tuple(sorted(set(values))) != values:
        raise ConfigurationError(f"{name} must be unique and strictly increasing")


def _strictly_increasing_floats(
    values: tuple[float, ...], name: str, *, allow_zero: bool = False
) -> None:
    minimum_ok = (lambda value: value >= 0) if allow_zero else (lambda value: value > 0)
    if not values or not all(minimum_ok(value) for value in values):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigurationError(f"{name} must contain {qualifier} values")
    if tuple(sorted(set(values))) != values:
        raise ConfigurationError(f"{name} must be unique and strictly increasing")


@dataclass(frozen=True)
class DataConfig:
    provider: str
    dataset: str
    schema: str
    symbols: tuple[str, ...]
    stype_in: str
    start: str
    end: str
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


@dataclass(frozen=True)
class ContractConfig:
    expected_tick_size: float
    expected_multiplier: float
    require_definition_match: bool


@dataclass(frozen=True)
class SessionConfig:
    timezone: str
    start_time: time
    end_time: time
    decision_grid_ms: int
    maximum_quote_age_ms: int


@dataclass(frozen=True)
class FeatureConfig:
    ofi_lookbacks_ms: tuple[int, ...]
    trade_lookbacks_ms: tuple[int, ...]
    event_intensity_lookback_ms: int
    lagged_return_ms: int
    depth_levels: tuple[int, ...]
    depth_weight_decay: float


@dataclass(frozen=True)
class LabelConfig:
    horizons_ms: tuple[int, ...]


@dataclass(frozen=True)
class SplitConfig:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date

    def split_for(self, session_date: date) -> str | None:
        if self.train_start <= session_date <= self.train_end:
            return "train"
        if self.validation_start <= session_date <= self.validation_end:
            return "validation"
        if self.holdout_start <= session_date <= self.holdout_end:
            return "holdout"
        return None


@dataclass(frozen=True)
class ExecutionConfig:
    primary_latency_ms: int
    latency_grid_ms: tuple[int, ...]
    primary_fee_per_contract_per_side_usd: float
    fee_grid_per_contract_per_side_usd: tuple[float, ...]
    primary_quantity: int
    quantity_grid: tuple[int, ...]
    maximum_open_positions: int


@dataclass(frozen=True)
class ResearchPlanConfig:
    ridge_alphas: tuple[float, ...]
    safety_margin_grid_ticks: tuple[float, ...]
    primary_safety_margin_ticks: float
    cv_folds: int
    minimum_train_sessions: int
    max_fit_rows_per_session: int
    bootstrap_repetitions: int


@dataclass(frozen=True)
class ResearchConfig:
    name: str
    seed: int
    data: DataConfig
    contract: ContractConfig
    session: SessionConfig
    features: FeatureConfig
    labels: LabelConfig
    splits: SplitConfig
    execution: ExecutionConfig
    research: ResearchPlanConfig
    source_path: Path


def _parse_time(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"invalid time: {value}") from exc


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"invalid date: {value}") from exc


def _require_sections(raw: dict[str, Any], sections: tuple[str, ...]) -> None:
    missing = [section for section in sections if section not in raw]
    if missing:
        raise ConfigurationError(f"missing configuration sections: {missing}")


def load_config(path: str | Path) -> ResearchConfig:
    """Load and validate the research configuration at *path*."""

    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")

    _require_sections(
        raw,
        (
            "project",
            "data",
            "contract",
            "session",
            "features",
            "labels",
            "splits",
            "execution",
            "research",
        ),
    )
    project = raw["project"]
    data = raw["data"]
    contract = raw["contract"]
    session = raw["session"]
    features = raw["features"]
    labels = raw["labels"]
    splits = raw["splits"]
    execution = raw["execution"]
    research = raw["research"]

    config = ResearchConfig(
        name=str(project["name"]),
        seed=int(project["seed"]),
        data=DataConfig(
            provider=str(data["provider"]),
            dataset=str(data["dataset"]),
            schema=str(data["schema"]),
            symbols=tuple(str(symbol) for symbol in data["symbols"]),
            stype_in=str(data["stype_in"]),
            start=str(data["start"]),
            end=str(data["end"]),
            raw_dir=Path(data["raw_dir"]),
            interim_dir=Path(data["interim_dir"]),
            processed_dir=Path(data["processed_dir"]),
        ),
        contract=ContractConfig(
            expected_tick_size=float(contract["expected_tick_size"]),
            expected_multiplier=float(contract["expected_multiplier"]),
            require_definition_match=bool(contract["require_definition_match"]),
        ),
        session=SessionConfig(
            timezone=str(session["timezone"]),
            start_time=_parse_time(str(session["start_time"])),
            end_time=_parse_time(str(session["end_time"])),
            decision_grid_ms=int(session["decision_grid_ms"]),
            maximum_quote_age_ms=int(session["maximum_quote_age_ms"]),
        ),
        features=FeatureConfig(
            ofi_lookbacks_ms=tuple(int(value) for value in features["ofi_lookbacks_ms"]),
            trade_lookbacks_ms=tuple(int(value) for value in features["trade_lookbacks_ms"]),
            event_intensity_lookback_ms=int(features["event_intensity_lookback_ms"]),
            lagged_return_ms=int(features["lagged_return_ms"]),
            depth_levels=tuple(int(value) for value in features["depth_levels"]),
            depth_weight_decay=float(features["depth_weight_decay"]),
        ),
        labels=LabelConfig(
            horizons_ms=tuple(int(value) for value in labels["horizons_ms"]),
        ),
        splits=SplitConfig(
            train_start=_parse_date(str(splits["train_start"])),
            train_end=_parse_date(str(splits["train_end"])),
            validation_start=_parse_date(str(splits["validation_start"])),
            validation_end=_parse_date(str(splits["validation_end"])),
            holdout_start=_parse_date(str(splits["holdout_start"])),
            holdout_end=_parse_date(str(splits["holdout_end"])),
        ),
        execution=ExecutionConfig(
            primary_latency_ms=int(execution["primary_latency_ms"]),
            latency_grid_ms=tuple(int(value) for value in execution["latency_grid_ms"]),
            primary_fee_per_contract_per_side_usd=float(
                execution["primary_fee_per_contract_per_side_usd"]
            ),
            fee_grid_per_contract_per_side_usd=tuple(
                float(value) for value in execution["fee_grid_per_contract_per_side_usd"]
            ),
            primary_quantity=int(execution["primary_quantity"]),
            quantity_grid=tuple(int(value) for value in execution["quantity_grid"]),
            maximum_open_positions=int(execution["maximum_open_positions"]),
        ),
        research=ResearchPlanConfig(
            ridge_alphas=tuple(float(value) for value in research["ridge_alphas"]),
            safety_margin_grid_ticks=tuple(
                float(value) for value in research["safety_margin_grid_ticks"]
            ),
            primary_safety_margin_ticks=float(research["primary_safety_margin_ticks"]),
            cv_folds=int(research["cv_folds"]),
            minimum_train_sessions=int(research["minimum_train_sessions"]),
            max_fit_rows_per_session=int(research["max_fit_rows_per_session"]),
            bootstrap_repetitions=int(research["bootstrap_repetitions"]),
        ),
        source_path=source_path,
    )
    validate_config(config)
    return config


def validate_config(config: ResearchConfig) -> None:
    """Enforce the v0.1 research contract."""

    if config.data.provider != "databento":
        raise ConfigurationError("v0.1 supports the Databento provider only")
    if config.data.schema != "mbp-10":
        raise ConfigurationError("v0.1 requires the mbp-10 schema")
    if not config.data.symbols:
        raise ConfigurationError("at least one exact raw symbol is required")
    if config.session.start_time >= config.session.end_time:
        raise ConfigurationError("v0.1 requires a non-overnight session window")
    if config.session.decision_grid_ms <= 0:
        raise ConfigurationError("decision_grid_ms must be positive")
    if config.session.maximum_quote_age_ms < config.session.decision_grid_ms:
        raise ConfigurationError("maximum_quote_age_ms cannot be smaller than the decision grid")
    if config.contract.expected_tick_size <= 0 or config.contract.expected_multiplier <= 0:
        raise ConfigurationError("contract tick size and multiplier must be positive")

    _strictly_increasing(config.features.ofi_lookbacks_ms, "ofi_lookbacks_ms")
    _strictly_increasing(config.features.trade_lookbacks_ms, "trade_lookbacks_ms")
    _strictly_increasing(config.features.depth_levels, "depth_levels")
    _strictly_increasing(config.labels.horizons_ms, "horizons_ms")
    if config.features.depth_levels[-1] > 10:
        raise ConfigurationError("MBP-10 cannot support depth levels above 10")

    split = config.splits
    if not (
        split.train_start
        <= split.train_end
        < split.validation_start
        <= split.validation_end
        < split.holdout_start
        <= split.holdout_end
    ):
        raise ConfigurationError("train, validation and holdout dates must be ordered and disjoint")

    if config.execution.primary_latency_ms not in config.execution.latency_grid_ms:
        raise ConfigurationError("primary latency must be included in the latency grid")
    if config.execution.primary_quantity not in config.execution.quantity_grid:
        raise ConfigurationError("primary quantity must be included in the quantity grid")
    if config.execution.maximum_open_positions != 1:
        raise ConfigurationError("v0.1 deliberately permits exactly one open position")

    _strictly_increasing_floats(config.research.ridge_alphas, "ridge_alphas")
    _strictly_increasing_floats(
        config.research.safety_margin_grid_ticks,
        "safety_margin_grid_ticks",
        allow_zero=True,
    )
    if config.research.primary_safety_margin_ticks not in config.research.safety_margin_grid_ticks:
        raise ConfigurationError("primary safety margin must be included in its grid")
    if config.research.cv_folds < 2:
        raise ConfigurationError("cv_folds must be at least 2")
    if config.research.minimum_train_sessions < 2:
        raise ConfigurationError("minimum_train_sessions must be at least 2")
    if config.research.max_fit_rows_per_session <= 0:
        raise ConfigurationError("max_fit_rows_per_session must be positive")
    if config.research.bootstrap_repetitions < 100:
        raise ConfigurationError("bootstrap_repetitions must be at least 100")
