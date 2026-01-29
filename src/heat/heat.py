"""
This module includes methods to simulate the conduction of heat across
an idealized metal plate.
"""
from dataclasses import dataclass
from enum import StrEnum

class Material(StrEnum):
    """StrEnum for different material types."""
    COPPER = "copper"

THERMAL_DIFFUSIVITY: dict[Material, float] = {
    Material.COPPER: 111.0
}
"""Mapping from Material to thermal diffusivity in m^2/s."""

@dataclass
class MetalPlateModel:
    """Dataclass that stores model parameters and states."""
    width: float = 0.05
    height: float = 0.05
    spatial_resolution: float = 0.001
    duration: float = 1.0
    temporal_resolution: float = 0.0001
    thermal_diffusivity: float = THERMAL_DIFFUSIVITY[Material.COPPER]
