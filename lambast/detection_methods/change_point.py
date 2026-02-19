import warnings
from dataclasses import dataclass
from itertools import cycle, pairwise
from typing import Any, Callable, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import ruptures
from numpy.typing import ArrayLike


@dataclass
class ChangePoint(object):
    """
    Class to detect changepoints
    """

    samples: ArrayLike | None = None
    detected: bool = False
    optimal_breakpoints: dict | None = None
    models: ClassVar[list[str]] = ["l1", "l2", "ar", "clinear", "cosine",
                                   "cosine", "linear", "mahalanobis", "normal",
                                   "rank", "rbf"]
    kernels: ClassVar[list[str]] = ["linear", "rbf", "cosine"]

    estimators: ClassVar[dict[str, Callable]] = {
        "Binseg": ruptures.Binseg,
        "BottomUp": ruptures.BottomUp,
        "Dynp": ruptures.Dynp,
        "KernelCPD": ruptures.KernelCPD,
        "Pelt": ruptures.Pelt,
        "Window": ruptures.Window
    }

    optional_args: ClassVar[list[str]] = [
        "model",
        "custom_cost",
        "min_size",
        "jump",
    ]

    choose_arg: ClassVar[dict[str, list[str]]] = {
        "Binseg": ["n_bkps", "pen", "epsilon"],
        "BottomUp": ["n_bkps", "pen", "epsilon"],
        "Dynp": ["n_bkps"],
        "KernelCPD": ["n_bkps", "pen"],
        "Pelt": ["pen"],
        "Window": ["n_bkps", "pen", "epsilon"],
    }

    def was_detected(self) -> bool:
        """
        Returns True if a changepoint was detected, False otherwise
        """

        return self.detected

    def set_samples(self, samples: ArrayLike) -> None:

        self.samples = np.array(samples)

    def get_estimators(self) -> str:
        """
        Retreive all possible estimators, with use parameters
        """

        s = "Estimators: "
        for key in ChangePoint.estimators:
            s += f"{key}: use with one of {ChangePoint.choose_arg[key]}\n"

        s += "--- Description of parameters ---\n"
        s += "n_bkps (int): number of breakpoints to find before stopping.\n"
        s += "penalty (float): penalty value (>0)\n"
        s += "epsilon (float): reconstruction budget (>0)"

        return s

    def get_models(self) -> str:
        """
        Retreive all possible models
        """

        s = f"Models: {ChangePoint.models}"
        return s

    def get_kernels(self) -> str:
        """
        Retreive all possible kernels
        """

        s = f"Kernels: {ChangePoint.kernels}"
        return s

    def detect_change_point(self, samples: ArrayLike | None = None,
                            estimator: str = "Pelt",
                            model: str = "rbf",
                            kernel: str = "rbf",
                            custom_cost: ruptures.base.BaseCost | None = None,
                            min_size: int = 2,
                            jump: int = 5,
                            **fit_args: dict[str, Any]) -> None:
        """
        Detect the change points

        Parameters:
        -samples: timeseries with potential changepoint
        -estimator: which estimator to use, choose one from get_estimators()
        -model: which cost function to use, choose one from get_models()
        -kernel: which kernel to use, choose one from get_kernels()
        -custom_cost: Custom cost function. Defaults to None.
        -min_size: Minimum segment length.
        -jump: subsample (one every *jump* points).
        -fit_args: a dictionary of parameters for the cost instance.
        """

        if samples is not None:
            self.samples = np.array(samples)
        assert type(self.samples) is np.ndarray

        self.optimal_breakpoints = None
        self.detected = False

        if self.samples is None or self.samples.shape[0] == 0:
            warnings.warn("No data, skipping detection")
            return

        est_args: dict[str, Any] = {"min_size": min_size, "jump": jump}
        if estimator == "KernelCPD":
            est_args["kernel"] = kernel
        else:
            est_args["model"] = model
            est_args["custom_cost"] = custom_cost

        self.optimal_breakpoints = ChangePoint.estimators[estimator](
            **est_args).fit(self.samples).predict(**fit_args)

        if len(self.optimal_breakpoints) > 1:
            self.detected = True

    def display(self, x_values: list[float] | None = None,
                save_fig: bool = True,
                figname: str = "test", filetype: str = "pdf",
                use_ax: plt.Axes | None = None,
                **plot_args) -> None:
        """
        Display the change points

        Optional parameters:
        -save_fig: whether to save the plotted figure to a file
        -figname: filename to save the plot, ignored if save_fig is false
        -filetype: savefile extension, ignored if save_fig is false
        -use_ax: specific plt.Axes to use, if used, assuming that figure will
            be displayed or saved elsewhere, so save_fig is ignored.
        """

        color_cycle = cycle(["#4286f4", "#f44174"])

        if self.samples is None:
            warnings.warn("No data to display")
            return

        if use_ax is None:
            if self.optimal_breakpoints is None:
                if x_values is None:
                    plt.plot(self.samples, **plot_args)
                else:
                    plt.plot(self.samples)
            else:
                ruptures.display(self.samples, self.optimal_breakpoints)
        else:
            if x_values is None:
                use_ax.plot(self.samples, **plot_args)
            else:
                use_ax.plot(x_values, self.samples, **plot_args)

            # Plot the breakpoints
            if self.optimal_breakpoints is not None:

                # NOTE:
                # This replicates the ruptures plotting of breakpoints
                # but allows us to use it in a multi-panel figure.
                bkps = [0] + list(self.optimal_breakpoints)
                for (start, end), col in zip(pairwise(bkps), color_cycle):
                    use_ax.axvspan(max(0, start - 0.5), end -
                                   0.5, facecolor=col, alpha=0.2)

        if use_ax is None and save_fig:
            plt.savefig(figname + '.' + filetype)
