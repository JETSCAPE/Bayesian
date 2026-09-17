from pathlib import Path

import pytest

from bayesian import analysis
from bayesian.emulation import base, interface


def _config(tmp_path: Path) -> dict:
    return {
        "observable_table_dir": str(tmp_path),
        "observable_config_dir": str(tmp_path),
        "observables_filename": "observables.h5",
        "output_dir": str(tmp_path / "results"),
        "analyses": {
            "analysis_test": {
                "parameterizations": ["unit"],
                "parameterization": {
                    "unit": {
                        "names": ["x"],
                        "min": [0.0],
                        "max": [1.0],
                    }
                },
                "parameters": {
                    "emulators": {
                        "hadron_group": {
                            "force_retrain": False,
                            "emulator_package": "sk_learn",
                            "n_pc": 1,
                            "kernels": {
                                "active": ["matern", "noise"],
                                "matern": {
                                    "nu": 1.5,
                                    "length_scale_bounds_factor": [0.1, 10],
                                },
                                "noise": {
                                    "type": "white",
                                    "args": {
                                        "noise_level": 0.1,
                                        "noise_level_bounds": [0.001, 10],
                                    },
                                },
                            },
                            "GPR": {"n_restarts": 1, "alpha": 1.0e-8},
                            "observable_list": ["hadron"],
                        },
                        "jet_group": {
                            "force_retrain": False,
                            "emulator_package": "sk_learn",
                            "n_pc": 1,
                            "kernels": {
                                "active": ["matern", "noise"],
                                "matern": {
                                    "nu": 1.5,
                                    "length_scale_bounds_factor": [0.1, 10],
                                },
                                "noise": {
                                    "type": "white",
                                    "args": {
                                        "noise_level": 0.1,
                                        "noise_level_bounds": [0.001, 10],
                                    },
                                },
                            },
                            "GPR": {"n_restarts": 1, "alpha": 1.0e-8},
                            "observable_list": ["jet"],
                        },
                    }
                },
            }
        },
    }


def _analysis_settings(tmp_path: Path, config: dict) -> analysis.AnalysisSettings:
    return analysis.AnalysisSettings.from_config(
        analysis_name="analysis_test",
        config_file=tmp_path / "config.yaml",
        config=config,
        parameterization="unit",
    )


def test_emulator_group_names_produce_distinct_output_filenames(tmp_path: Path) -> None:
    config = _config(tmp_path)
    analysis_settings = analysis.AnalysisSettings.from_config(
        analysis_name="analysis_test",
        config_file=tmp_path / "config.yaml",
        config=config,
        parameterization="unit",
    )

    emulation_config = interface.EmulationConfig.from_config_file(analysis_settings)
    filenames = {
        group_name: base.IO.output_filename(settings, analysis_settings).name
        for group_name, settings in emulation_config.emulation_settings.items()
    }

    assert filenames == {
        "hadron_group": "emulator_hadron_group.pkl",
        "jet_group": "emulator_jet_group.pkl",
    }


def test_single_unnamed_group_preserves_legacy_filename(tmp_path: Path) -> None:
    config = _config(tmp_path)
    groups = config["analyses"]["analysis_test"]["parameters"]["emulators"]
    groups.pop("jet_group")
    analysis_settings = _analysis_settings(tmp_path, config)

    emulation_config = interface.EmulationConfig.from_config_file(
        analysis_settings
    )
    settings = emulation_config.emulation_settings["hadron_group"]

    assert base.IO.output_filename(settings, analysis_settings).name == "emulator.pkl"


@pytest.mark.parametrize("additional_name", ["shared", ""])
def test_duplicate_explicit_emulator_names_are_rejected(
    tmp_path: Path,
    additional_name: str,
) -> None:
    config = _config(tmp_path)
    groups = config["analyses"]["analysis_test"]["parameters"]["emulators"]
    for group in groups.values():
        group["additional_name"] = additional_name
    analysis_settings = _analysis_settings(tmp_path, config)

    with pytest.raises(ValueError, match="resolve to"):
        interface.EmulationConfig.from_config_file(analysis_settings)
