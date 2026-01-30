"""
This module includes methods to simulate the conduction of heat across
an idealized plate made of uniform material.

Classes
=======
UniformPlateModel
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import Self
import warnings

import numpy as np
import numpy.typing as npt

class NonPhysicalValueError(Exception):
    """Exception raised for non-physical dimensions."""

class NumericalInstabilityError(Exception):
    """Exception raised for numerical instability."""

class UnknownMaterialError(Exception):
    """Exception raised for unknown plate materials."""

class Material(StrEnum):
    """StrEnum for different material types."""
    COPPER = "copper"

THERMAL_DIFFUSIVITY: dict[Material, float] = {
    Material.COPPER: 0.000111
}
"""Mapping from Material to thermal diffusivity in m^2/s."""

@dataclass
class UniformPlateModel:
    """
    Dataclass that stores model parameters and states for a finite difference
    simulation of the heat equation. Model parameters assume a plate made of
    uniform material with a constant heat source along the left edge. The right,
    top, and bottom boundaries of the plate are perfect insulators. The
    simulation assumes the thermal diffusivity of the material is constant
    with temperature.
    
    Attributes
    ----------
    width: float, default 0.15
        Plate width in meters.
    height: float, default 0.15
        Plate height in meters.
    spatial_resolution: float, default 0.003
        Horizontal and vertical grid cell spacing in meters.
    duration: float, default 1.0
        Total simulation duration in seconds.
    temporal_resolution: float, default 0.01
        Model time step in seconds.
    material: Material, default Material.COPPER
        Assumed plate material used to determine thermal diffusivity.
    initial_temperature: float, default 300.0
        Initial uniform temperature of plate in Kelvin.
    left_boundary_temperature: float, default 400.0
        Temperature of left boundary constant heat source in Kelvin.
    write_interval: float, default 0.1
        Interval in seconds at which to save intermediate states to self.history.
    history: dict[str, npt.NDArray[np.float64]]
        Mapping from a unique key indicating elapsed model time to a saved model state.
    state: npt.NDArray[np.float64]
        Current state of plate temperature field.
    current_time: float
        Current elapsed model time in seconds.
    
    Raises
    ------
    NonPhysicalValueError, NumericalInstabilityError, UnknownMaterialError
    """
    width: float = 0.15
    height: float = 0.15
    spatial_resolution: float = 0.003
    duration: float = 1.0
    temporal_resolution: float = 0.01
    material: Material = Material.COPPER
    initial_temperature: float = 300.0
    left_boundary_temperature: float = 400.0
    write_interval: float | None = 0.1

    def __post_init__(self: Self) -> None:
        # Validate spatial dimensions
        if self.width <= 0.0:
            raise NonPhysicalValueError(f"width {self.width} is not >= 0")
        if self.height <= 0.0:
            raise NonPhysicalValueError(f"height {self.height} is not >= 0")
        if self.spatial_resolution <= 0.0:
            raise NonPhysicalValueError(
                f"spatial_resolution "
                f"{self.spatial_resolution} is not >= 0"
            )
        if self.spatial_resolution > self.width:
            raise NonPhysicalValueError(
                f"spatial_resolution {self.spatial_resolution}"
                f" must be <= width {self.width}"
            )
        if self.spatial_resolution > self.height:
            raise NonPhysicalValueError(
                f"spatial_resolution {self.spatial_resolution}"
                f" must be <= height {self.height}"
            )

        # Validate temporal dimensions
        if self.duration < 0.0:
            raise NonPhysicalValueError(f"duration {self.duration} is not > 0")
        if self.temporal_resolution <= 0.0:
            raise NonPhysicalValueError(
                f"temporal_resolution {self.temporal_resolution} is not >= 0"
            )
        if self.temporal_resolution > self.duration:
            raise NonPhysicalValueError(
                f"temporal_resolution {self.temporal_resolution}"
                f" must be <= duration {self.duration}"
            )

        # Compute number of time steps and initialize current time step
        self._current_time: float = 0.0
        self._time_steps = int(self.duration / self.temporal_resolution)

        # Validate state history parameters
        self.history: dict[str, npt.NDArray[np.float64]] = {}
        if self.write_interval is not None:
            if self.write_interval < self.temporal_resolution:
                raise NonPhysicalValueError(
                    f"write_interval {self.write_interval} must be >="
                    f" temporal_resolution {self.temporal_resolution}"
                )

            # Convert write_interval from seconds to time steps
            self._write_steps = int(self.write_interval / self.temporal_resolution)
        else:
            # Set _write_steps to a large value
            self._write_steps = self._time_steps * 2

        # Validate material
        if self.material not in THERMAL_DIFFUSIVITY:
            raise UnknownMaterialError(
                f"unknown material '{self.material}'"
                f", must be one of {list(THERMAL_DIFFUSIVITY.keys())}"
            )

        # Validate temperatures
        if self.initial_temperature < 0.0:
            raise NonPhysicalValueError(
                f"initial_temperature {self.initial_temperature} is not > 0"
            )
        if self.left_boundary_temperature < 0.0:
            raise NonPhysicalValueError(
                f"left_boundary_temperature {self.left_boundary_temperature} is not > 0"
            )

        # Warn for 0 duration
        if self.duration == 0.0:
            warnings.warn("duration is 0", UserWarning)

        # Warn for non-dynamic simulation
        if self.initial_temperature == self.left_boundary_temperature:
            warnings.warn(
                f"initial_temperature {self.initial_temperature} "
                f"and left_boundary_temperature {self.left_boundary_temperature} are equal"
            )

        # Set thermal diffusivity
        self._thermal_diffusivity = THERMAL_DIFFUSIVITY[self.material]

        # Compute the Fourier coefficient (AKA the diffusion number)
        self._diffusion = (
            self._thermal_diffusivity * self.temporal_resolution /
            (self.spatial_resolution ** 2.0)
        )

        # Warn for instability
        #   For a finite difference numerical scheme with a uniform grid, the
        #   Fourier coefficient must be <= 0.25 for numerical stability
        if self._diffusion > 0.25:
            raise NumericalInstabilityError(
                f"Diffusion number {self._diffusion:.3f} > 0.25"
                " simulation may be numerically unstable"
            )

        # Compute dimensions of plate in number of grid cells
        self._columns = int(self.width / self.spatial_resolution)
        self._rows = int(self.height / self.spatial_resolution)

        # Initialize internal state with a buffer to enforce boundary conditions
        self._state = np.full((self._columns+2, self._rows+2), self.initial_temperature)

        # Set left boundary condition
        self._state[0, :] = self.left_boundary_temperature

    def update_state(self: Self) -> None:
        """Update the state for a single time step."""
        # Update state
        old_state = self._state.copy()
        for i in range(1, self._columns+1):
            for j in range(1, self._rows+1):
                self._state[i, j] = (
                    (self._diffusion * (
                        old_state[i+1, j] - 2.0 * old_state[i, j] + old_state[i-1, j]
                    )) +
                    (self._diffusion * (
                        old_state[i, j+1] - 2.0 * old_state[i, j] + old_state[i, j-1]
                    )) +
                    old_state[i, j]
                )

    def enforce_boundary_conditions(self: Self) -> None:
        """Enforce simulation boundary conditions."""
        # Enforce boundary conditions
        self._state[:, -1] = self._state[:, -2] # Top
        self._state[:, 0] = self._state[:, 1] # Bottom
        self._state[-1, :] = self._state[-2, :] # Right
        self._state[0, :] = self.left_boundary_temperature # Left

    def run(self: Self) -> None:
        """Run the simulation for full duration."""
        # Save initial state
        self.save_state()

        # March in time
        for step in range(1, self._time_steps+1):
            # Advance elapsed time
            self._current_time += self.temporal_resolution

            # Update model state
            self.update_state()

            # Enforce boundary conditions
            self.enforce_boundary_conditions()

            # Save state
            if step % self._write_steps == 0:
                self.save_state()

        # Save final state
        self.save_state()

    def save_state(self: Self) -> None:
        """If write_interval set, add current model state to history."""
        # Save final state
        if self.write_interval is not None:
            key = f"Model time: {self.current_time:.2f} s"
            self.history[key] = self.state.copy()

    @property
    def state(self: Self) -> npt.NDArray[np.float64]:
        """Current state of plate temperature field."""
        # Return inner cells without boundary condition buffer
        #   Transpose for easier plotting
        return self._state[1:-1, 1:-1].T

    @property
    def current_time(self: Self) -> float:
        """Returns current elapsed model time, accounting for temporal precision."""
        # Determine precision of temporal resolution
        decimals = int(np.round(np.log10(self.temporal_resolution)*-1))

        # Drop extra digits from machine precision
        return np.round(self._current_time, decimals)
