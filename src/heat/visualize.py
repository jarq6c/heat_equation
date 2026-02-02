"""
This module includes methods to visualize model state.

Classes
=======
Frame
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
    """
    Dataclass that stores frame data for animations.

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Model data.
    time: np.float64
        Model elapsed time that corresponds to data.
    """
    data: npt.NDArray[np.float64]
    time: np.float64

class FieldPlotter:
    """
    Configure and render plots of field data.

    Parameters
    ----------
    value_label: str
    data_shape: tuple[float, float]
    domain_extent: tuple[float, float, float, float]
    value_range: tuple[float, float]
    figure_size: tuple[float, float]
    x_label: str
    y_label: str
    time_units: str, default "s"

    Attributes
    ----------
    figure
    axes
    image
    time_units: str, default "s"
    """
    def __init__(
            self: Self,
            value_label: str,
            data_shape: tuple[float, float],
            domain_extent: tuple[float, float, float, float],
            value_range: tuple[float, float],
            figure_size: tuple[float, float],
            x_label: str,
            y_label: str,
            time_units: str = "s"
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
        self.time_units = time_units
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

    def update_data(
            self: Self,
            data: npt.NDArray[np.float64]
            ) -> AxesImage:
        """
        Replace figure values with new field.

        Parameters
        ----------
        data: npt.NDArray[np.float64]
            Replacement data.
        
        Returns
        -------
        _ : AxesImage
            New image.
        """
        self.image.set_array(data)
        return self.image

    def update_title(
            self: Self,
            figure_title: str
            ) -> None:
        """
        Replace figure title.

        Parameters
        ----------
        figure_title: str
            Replacement string for title.
        """
        self.axes.set_title(figure_title)

    def to_png(
            self: Self,
            data: npt.NDArray[np.float64],
            output: Path
        ) -> None:
        """
        Save data to PNG file.

        Parameters
        ----------
        data: npt.NDArray[np.float64]
            Data to plot.
        output: Path
            Output path for PNG file.
        """
        # Update data
        self.update_data(data=data)

        # Save figure
        self.figure.savefig(output)

    def to_gif(
            self: Self,
            frames: list[tuple[np.float64, npt.NDArray[np.float64]]],
            output: Path,
            frames_per_second: int = 30
        ) -> None:
        """
        Save sequence of data to animated GIF.

        Parameters
        ----------
        frames: list[tuple[np.float64, npt.NDArray[np.float64]]]
            List of tuples where the first element is the elapsed model time and
            the second element is the model state. States are plotted in order.
        output: Path
            Output path for GIF file.
        frames_per_second: int, default 30
            Frame rate of resulting GIF.
        """
        # Simplify frames
        frame_list = [Frame(data=d, time=t) for t, d in frames]

        # Make frame update function
        def updater(frame: Frame) -> list[AxesImage]:
            self.update_title(f"Model time: {frame.time:.2f} ({self.time_units})")
            return [self.update_data(frame.data)]

        # Generate animation
        animation = FuncAnimation(self.figure, updater, frames=frame_list, blit=True)

        # Save
        animation.save(
            output,
            writer='pillow',
            fps=frames_per_second
        )
