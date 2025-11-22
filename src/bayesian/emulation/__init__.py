"""Emulation module for Bayesian Inference.

This module provides functionality to train and call emulators to replace expensive
forward model evaluations.

Emulation is handled through the `EmulationConfig` class. The concept is that
you can configure one or more emulators to provide emulation of the expensive
forward model. You use multiple emulators if you want:
- Different emulators for different observables. e.g. one devoted to hadron RAA,
  and another devoted to jet RAA.
- You want to use different packages to perform emulation.

Using the EmulationConfig, there are two main functionalities:
 - fit_emulator(), which trains the emulator(s) on the provided data.
 - predict() construct mean, std dev of emulator(s) for a given set of parameter values.

For further information, see the documentation in `emulation.interface`

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

from bayesian.emulation.interface import (  # noqa: F401
    EmulationConfig,
    fit_emulators,
    predict,
)
