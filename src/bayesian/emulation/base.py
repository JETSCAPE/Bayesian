"""Base functionality needed for implementing an emulator.

This is **NOT** for the user interface, but rather for developers specifying
how to interact with an individual emulator. The user interface is implemented
in `interface`

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, ClassVar, Protocol

import attrs

from bayesian import analysis, data_IO

logger = logging.getLogger(__name__)


@attrs.define
class BaseEmulatorSettings:
    """Base (i.e. shared) settings for an emulator.

    Store this class in your specialized emulator settings class.
    Composition is preferred to inheritance.

    Attributes:
        force_retrain: If true, force the emulator to retrain.
        _settings: YAML config corresponding to the emulator settings.
            It's included here since we need access for derived properties, but it's marked
            as private since we want to encourage access through the specialized emulator settings.

    """

    # emulator_package: str
    # analysis_name: str
    # parameterization: str
    # config_file: Path = attrs.field(converter=Path)
    # analysis_config: dict[str, Any] = attrs.field(factory=dict)
    # emulation_group_name: str | None = None  # <-- optional, passed from higher-level config
    # config: dict[str, Any] = attrs.field(init=False)
    # observables_table_dir: Path | str = attrs.field(init=False)
    # observables_config_dir: Path | str = attrs.field(init=False)
    # observables_filename: str = attrs.field(init=False)
    # emulation_outputfile: Path = attrs.field(init=False)
    # TODO(RJE): Starting actual settings here. Others should be passed in separately, I think...
    force_retrain: bool = attrs.field()
    _settings: dict[str, Any] = attrs.field()
    # TODO(RJE): Does this really belong here? Not sure...
    _observable_filter: data_IO.ObservableFilter | None = attrs.field(init=False)

    # def __attrs_post_init__(self):
    #     """
    #     Post-creation customization of the emulator configuration.
    #     """
    #     with Path(self.config_file).open() as stream:
    #         config = yaml.safe_load(stream)

    #     # Observable inputs
    #     self.config = config
    #     self.observables_table_dir = config["observable_table_dir"]
    #     self.observables_config_dir = config["observable_config_dir"]
    #     self.observables_filename = config["observables_filename"]

    #     # Build the output directory
    #     output_dir = Path(config["output_dir"]) / f"{self.analysis_name}_{self.parameterization}"

    #     # Choose file name based on group name
    #     if self.emulation_group_name:
    #         emulation_outputfile_name = f"emulation_{self.emulation_group_name}.pkl"
    #     else:
    #         emulation_outputfile_name = "emulation.pkl"

    #     self.emulation_outputfile = output_dir / emulation_outputfile_name

    # @classmethod
    # def from_config(cls, config: dict[str, Any]) -> BaseEmulatorSettings:
    #     """
    #     Initialize the emulator configuration from a config file.
    #     """
    #     c = cls(
    #         emulator_name=config["emulator_name"],
    #         analysis_name=config["analysis_name"],
    #         parameterization=config["parameterization"],
    #         config_file=config["config_file"],
    #         emulation_group_name=config.get("emulation_group_name"),
    #     )
    #     return c

    @classmethod
    def from_emulator_settings(cls, emulator_settings: dict[str, Any]) -> BaseEmulatorSettings:
        """Initialize the base emulator settings from a emulator settings."""
        return cls(
            force_retrain=emulator_settings["force_retrain"],
            settings=emulator_settings,
        )

    @property
    def observable_filter(self) -> data_IO.ObservableFilter | None:
        if self._observable_filter is not None:
            return self._observable_filter
        # Observable filter
        self._observable_filter = None
        observable_list = self._settings.get("observable_list", [])
        observable_exclude_list = self._settings.get("observable_exclude_list", [])
        if observable_list or observable_exclude_list:
            self._observable_filter = data_IO.ObservableFilter(
                include_list=observable_list,
                exclude_list=observable_exclude_list,
            )
        return self.observable_filter


class EmulatorSettings(Protocol):
    """
    Protocol for an emulator settings.
    """

    emulator_name: ClassVar[str]
    base_settings: BaseEmulatorSettings
    settings: dict[str, Any]
    # More specific name
    additional_name: str = ""


@attrs.define
class IO:
    @staticmethod
    def output_filename(emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings) -> Path:
        filename = "emulator.pkl"
        if emulator_settings.additional_name:
            filename = f"emulator_{emulator_settings.additional_name}.pkl"
        return analysis_settings.output_dir / filename

    @staticmethod
    def read_emulator(emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings) -> Any:
        """
        Read emulators from file.
        """
        filename = IO.output_filename(emulator_settings=emulator_settings, analysis_settings=analysis_settings)

        with filename.open("rb") as f:
            results: dict[str, Any] = pickle.load(f)
        return results["emulator"]

    @staticmethod
    def write_emulator(
        emulator: Any, emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings
    ) -> None:
        """
        Write emulators stored in a result from `fit_emulator_group` to file.
        """
        filename = IO.output_filename(emulator_settings=emulator_settings, analysis_settings=analysis_settings)

        with filename.open("wb") as f:
            pickle.dump({"emulator": emulator}, f)
