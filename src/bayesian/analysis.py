"""Primary analysis parameters.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import yaml

logger = logging.getLogger(__name__)


@attrs.define
class AnalysisIO:
    observables_table_dir: Path | str = attrs.field(converter=Path)
    observables_config_dir: Path | str = attrs.field(converter=Path)
    observables_filename: str = attrs.field()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AnalysisIO:
        return cls(
            observables_table_dir=config["observable_table_dir"],
            observables_config_dir=config["observable_config_dir"],
            observables_filename=config["observables_filename"],
        )

    @classmethod
    def from_config_file(cls, config_file: str | Path) -> AnalysisIO:
        with Path(config_file).open() as stream:
            config = yaml.safe_load(stream)

        return cls.from_config(config=config)


@attrs.define
class AnalysisConfig:
    name: str
    parametrization: str
    config_file: Path = attrs.field(converter=Path)
    io: AnalysisIO
    raw_analysis_config: dict[str, Any] = attrs.field(factory=dict)

    @classmethod
    def from_config(cls, config_file: Path, config: dict[str, Any]) -> AnalysisConfig:
        """
        Initialize the analysis configuration from a config file.
        """
        return cls(
            name=config["analysis_name"],
            parametrization=config["parameterization"],
            config_file=config_file,
            io=AnalysisIO.from_config(config=config),
            raw_analysis_config=config,
        )

    @classmethod
    def from_config_file(cls, config_file: str | Path) -> AnalysisConfig:
        with Path(config_file).open() as stream:
            config = yaml.safe_load(stream)

        return cls.from_config(config_file=Path(config_file), config=config)
