"""Fixed, original-style standard plot report for the public pipeline."""

from __future__ import annotations

from pathlib import Path

from bioloc._kernels import plotting as legacy

from .stages import (
    plot_association_stage,
    plot_histogram_stage,
    plot_local_tdoa_stage,
    plot_localization_stage,
    plot_reconstruction_stage,
    plot_roi_stage,
)


def make_standard_plots(recording, result, config, *, show: bool, save: bool):
    """Create the fixed set of scientifically useful original-style figures."""
    import matplotlib.pyplot as plt

    figures = []
    figures.extend(plot_roi_stage(recording, result.roi, config, show=False))
    figures.extend(plot_local_tdoa_stage(recording, result.local_tdoas, config, show=False))
    figures.extend(
        plot_histogram_stage(result.peaks, result.local_tdoas, config, show=False)
    )
    figures.extend(
        plot_association_stage(
            result.peaks,
            result.local_tdoas,
            result.association,
            config,
            show=False,
        )
    )
    figures.extend(
        plot_reconstruction_stage(
            recording,
            result.local_tdoas,
            result.association,
            result.reconstruction,
            config,
            show=False,
        )
    )
    figures.extend(
        plot_localization_stage(
            recording,
            result.association,
            result.localization,
            config,
            show=False,
        )
    )
    if save:
        output_directory = Path("Figures") / "BioLoc"
        output_directory.mkdir(parents=True, exist_ok=True)
        for figure, filename in figures:
            legacy.savefig(
                figure,
                str(output_directory / filename),
                save_pdf=False,
                save_png=True,
                close=not show,
            )
    if show:
        plt.show()
    elif not save:
        for figure, _ in figures:
            plt.close(figure)
    return [figure for figure, _ in figures]


__all__ = ["make_standard_plots"]
