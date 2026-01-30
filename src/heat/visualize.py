"""
This module includes methods to visualize model state.

Classes
=======
FieldPlotter
"""
from typing import Self
from pathlib import Path

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

class FieldPlotter:
    """A dataclass for storing plot parameters."""
    def __init__(
            self: Self,
            data: npt.NDArray[np.float64],
            value_label: str,
            domain_extent: tuple[float, float, float, float],
            value_range: tuple[float, float],
            figure_size: tuple[float, float],
            figure_title: str,
            x_label: str,
            y_label: str
            ) -> None:
        # Set up figure and axes
        self.figure, self.axes = plt.subplots(figsize=figure_size)

        # Instantiate image
        self.image = self.axes.imshow(
            data,
            cmap="hot",
            origin="lower",
            extent=domain_extent,
            vmin=value_range[0],
            vmax=value_range[1]
        )

        # Add colorbar
        self.figure.colorbar(self.image, ax=self.axes, label=value_label)

        # Set labels
        self.axes.set_title(figure_title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

    def update(
            self: Self,
            data: npt.NDArray[np.float64],
            figure_title: str | None = None
            ) -> AxesImage:
        """Update figure data."""
        # Update data
        self.image.set_array(data)

        # Update figure title
        if figure_title is not None:
            self.axes.set_title(figure_title)

        return self.image

    def to_png(self: Self, output: Path) -> None:
        """Save current figure to PNG."""
        self.figure.savefig(output)
