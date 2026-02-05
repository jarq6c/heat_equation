"""Tests for the model module."""
import pytest
import numpy as np
from src.heat.model import (UniformPlateModel, Material, NonPhysicalValueError,
    UnknownMaterialError, NumericalInstabilityError)

@pytest.fixture
def default_model() -> UniformPlateModel:
    """Returns a model with default parameters."""
    return UniformPlateModel(
        width=0.15,
        height=0.15,
        spatial_resolution=0.003,
        duration=1.0,
        temporal_resolution=0.01,
        material=Material.COPPER,
        initial_temperature=300.0,
        left_boundary_temperature=400.0,
        write_interval=0.1,
        error_warning_threshold=1e-06
    )

@pytest.fixture
def single_step_model() -> UniformPlateModel:
    """Returns a model that runs for a single time step."""
    return UniformPlateModel(duration=0.01, temporal_resolution=0.01)

def test_defaults(default_model) -> None:
    """Test model defaults."""
    assert default_model == UniformPlateModel()

@pytest.mark.parametrize(
    "parameter_name, parameter_value",
    [
        ("width", -0.15),
        ("height", -0.15),
        ("spatial_resolution", -0.003),
        ("spatial_resolution", 100.0),
        ("temporal_resolution", -1.0),
        ("duration", -1.0),
        ("initial_temperature", -1.0),
        ("left_boundary_temperature", -1.0),
        ("write_interval", -1.0)
    ]
)
def test_nonphysical_errors(parameter_name, parameter_value) -> None:
    """Test that model correctly raises NonPhysicalValueError."""
    with pytest.raises(NonPhysicalValueError):
        UniformPlateModel(**{parameter_name: parameter_value})

def test_unknown_material_error() -> None:
    """Test that model correct raises for unknown materials."""
    with pytest.raises(UnknownMaterialError):
        UniformPlateModel(material="unobtainium")

@pytest.mark.parametrize(
    "parameters",
    [
        {
            "duration": 60.0,
            "write_interval": 2.0,
            "spatial_resolution": 0.00075,
            "temporal_resolution": 1.0
        },
    ]
)
def test_numerical_instability_errors(parameters) -> None:
    """Test that model correctly raises NumericalInstabilityError."""
    with pytest.raises(NumericalInstabilityError):
        UniformPlateModel(**parameters)

def test_model_step(single_step_model) -> None:
    """Test single model step."""
    diffusion = 0.000111 * 0.01 / (0.003 ** 2.0)
    old_state = np.full((50, 50), 300.0)
    new_state = np.empty((50, 50), dtype=np.float64)
    for i in range(50):
        for j in range(50):
            if i == 0:
                left = 400.0
            else:
                left = old_state[i-1, j]

            if i == 49:
                right = 300.0
            else:
                right = old_state[i+1, j]

            if j == 0:
                bottom = 300.0
            else:
                bottom = old_state[i, j-1]

            if j == 49:
                top = 300.0
            else:
                top = old_state[i, j+1]

            new_state[i, j] = (
                (diffusion * (
                    right - 2.0 * old_state[i, j] + left
                )) +
                (diffusion * (
                    top - 2.0 * old_state[i, j] + bottom
                )) +
                old_state[i, j]
            )

    model = single_step_model
    model.run()
    assert np.all(model.state == new_state.T)

# @pytest.mark.slow
def test_full_simulation(default_model) -> None:
    """Test full simulation."""
    diffusion = 0.000111 * 0.01 / (0.003 ** 2.0)
    old_state = np.full((50, 50), 300.0)
    new_state = np.empty((50, 50), dtype=np.float64)
    for _ in range(100):
        for i in range(50):
            for j in range(50):
                if i == 0:
                    left = 400.0
                else:
                    left = old_state[i-1, j]

                if i == 49:
                    right = old_state[i, j]
                else:
                    right = old_state[i+1, j]

                if j == 0:
                    bottom = old_state[i, j]
                else:
                    bottom = old_state[i, j-1]

                if j == 49:
                    top = old_state[i, j]
                else:
                    top = old_state[i, j+1]

                new_state[i, j] = (
                    (diffusion * (
                        right - 2.0 * old_state[i, j] + left
                    )) +
                    (diffusion * (
                        top - 2.0 * old_state[i, j] + bottom
                    )) +
                    old_state[i, j]
                )
        old_state = new_state.copy()

    model = default_model
    model.run()
    assert np.all(model.state == new_state.T)
