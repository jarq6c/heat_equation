"""
Build and run a model that simulates the conduction of heat across a
uniform material.
"""
from src.heat.model import UniformPlateModel
from src.heat.visualize import FieldPlotter

model = UniformPlateModel(duration=10.0, write_interval=0.5)
model.run()

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
