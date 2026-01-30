"""
This module includes methods to simulate the conduction of heat across
an idealized plate made of uniform material.

Classes
=======
PlateModel
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import Self
import warnings

import numpy as np

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
    
    Raises
    ------
    NonPhysicalValueError
    """
    width: float = 0.15
    height: float = 0.15
    spatial_resolution: float = 0.003
    duration: float = 1.0
    temporal_resolution: float = 0.01
    material: Material = Material.COPPER
    initial_temperature: float = 300.0
    left_boundary_temperature: float = 400.0

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
        #   Fourier coefficient must be >= 0.25 for numerical stability
        if self._diffusion > 0.25:
            raise NumericalInstabilityError(
                f"Diffusion number {self._diffusion:.3f} > 0.25"
                " simulation may be numerically unstable"
            )

        # Compute dimensions of plate in number of grid cells
        nx = int(self.width / self.spatial_resolution) # horizontal cells
        ny = int(self.height / self.spatial_resolution) # vertical cells

        # Initialize state (columns, rows)
        self._state = np.full((nx, ny), self.initial_temperature)
