"""
This module includes methods to visualize model state.

Classes
=======
FieldPlotter
"""
from typing import Self
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.animation import FuncAnimation

@dataclass
class Frame:
    """Dataclass that stores frame data."""
    data: npt.NDArray[np.float64]
    time: np.float64

class FieldPlotter:
    """A dataclass for storing plot parameters."""
    def __init__(
            self: Self,
            value_label: str,
            data_shape: tuple[float, float],
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
            np.empty(shape=data_shape, dtype=np.float64),
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

    def update_data(
            self: Self,
            data: npt.NDArray[np.float64]
            ) -> AxesImage:
        """Update figure data."""
        self.image.set_array(data)
        return self.image

    def update_title(
            self: Self,
            figure_title: str
            ) -> AxesImage:
        """Update figure title."""
        self.axes.set_title(figure_title)

    def to_png(
            self: Self,
            data: npt.NDArray[np.float64],
            output: Path
        ) -> None:
        """Save current figure to PNG."""
        # Update data
        self.update_data(data=data)

        # Save figure
        self.figure.savefig(output)

    def to_gif(
            self: Self,
            frames: dict[str, npt.NDArray[np.float64]],
            output: Path,
            frames_per_second: int = 30
        ) -> None:
        """Save sequence of frames to animated GIF."""
        # Simplify frames
        frame_list = [Frame(data=d, time=t) for t, d in frames.items()]

        # Make frame update function
        def updater(frame: Frame) -> list[AxesImage]:
            self.update_title(f"Model time: {frame.time:.2f}")
            return [self.update_data(frame.data)]

        # Generate animation
        animation = FuncAnimation(self.figure, updater, frames=frame_list, blit=True)

        # Save
        animation.save(
            output,
            writer='pillow',
            fps=frames_per_second
        )
