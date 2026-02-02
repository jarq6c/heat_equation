"""
Build and run a model that simulates the conduction of heat across a
uniform material.
"""
from src.heat.model import UniformPlateModel
from src.heat.visualize import FieldPlotter

# Instantiate and run simulation
model = UniformPlateModel(
    duration=60.0,
    write_interval=2.0,
    # spatial_resolution=0.0003,
    # temporal_resolution=0.0001
    )
model.run()

print(f"Error: {model.balance_error}")

# Render state history as animate GIF
plotter = FieldPlotter(
    data_shape=model.state.shape,
    value_label="Temperature (K)",
    domain_extent=(0, model.width, 0, model.height),
    value_range=(model.initial_temperature, model.left_boundary_temperature),
    figure_size=(6, 5),
    x_label="Horizontal (m)",
    y_label="Vertical (m)"
)
plotter.to_gif(model.history, "heat_equation.gif", 10)
