from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from process_manager import (
    BernoulliDistribution,
    CategoricalDistribution,
    DistName,
    DistType,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    PoissonDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)

ASSET_DIR = Path(__file__).parent.parent / "docs" / "assets" / "distributions"
ORANGE = "#FF5F05"
BLUE = "#13294B"

DPI = 100
ASPECT_RATIO = 12 / 5

WIDTH = 600 / DPI
HEIGHT = WIDTH / ASPECT_RATIO


def get_dist(
    dist_type: DistType,
) -> tuple[
    BernoulliDistribution
    | CategoricalDistribution
    | ExponentialDistribution
    | LogNormalDistribution
    | NormalDistribution
    | PoissonDistribution
    | TriangularDistribution
    | TruncatedNormalDistribution
    | UniformDistribution,
    str,
]:

    name = DistName(dist_type)

    sup_title = f"{name.replace('_', ' ').title()} Distribution "

    match dist_type:
        case DistType.NORMAL:
            mu = 0
            sigma = 1
            dist = NormalDistribution(name=name, mu=mu, sigma=sigma)
            sup_title += f"({mu=} {sigma=})"
        case DistType.UNIFORM:
            low = -1
            high = 2
            dist = UniformDistribution(name=name, low=low, high=high)
            sup_title += f"({low=} {high=})"
        case DistType.CATEGORICAL:

            class Blood(StrEnum):
                O_P = "O+"
                O_N = "O-"
                A_P = "A+"
                A_N = "A-"
                B_P = "B+"
                B_N = "B-"
                AB_P = "AB+"
                AB_N = "AB-"

            dist = CategoricalDistribution[Blood](
                name=name,
                choices={
                    Blood.O_P: 0.36,
                    Blood.O_N: 0.14,
                    Blood.A_P: 0.28,
                    Blood.A_N: 0.08,
                    Blood.B_P: 0.08,
                    Blood.B_N: 0.03,
                    Blood.AB_P: 0.02,
                    Blood.AB_N: 0.01,
                },
                nominal=Blood.O_P,
            )
            sup_title += "(blood types)"
        case DistType.TRIANGULAR:
            low = 0
            high = 1
            mode = 0.75
            dist = TriangularDistribution(name=name, low=low, high=high, mode=mode)
            sup_title += f"({low=} {high=} {mode=})"
        case DistType.TRUNCATED_NORMAL:
            mu = 1
            sigma = 1
            low = 0
            dist = TruncatedNormalDistribution(name=name, mu=mu, sigma=sigma, low=low)
            sup_title += f"({mu=} {sigma=} {low=})"
        case DistType.LOG_NORMAL:
            s = 1
            scale = 1
            dist = LogNormalDistribution(name=name, s=s, scale=scale)
            sup_title += f"({s=} {scale=})"
        case DistType.POISSON:
            lam = 4
            dist = PoissonDistribution(name=name, lam=lam)
            sup_title += f"({lam=})"
        case DistType.EXPONENTIAL:
            lam = 1
            dist = ExponentialDistribution(name=name, lam=lam)
            sup_title += f"({lam=})"
        case DistType.BERNOULLI:
            p = 0.75
            dist = BernoulliDistribution(name=name, p=p)
            sup_title += f"({p=})"
        case _:
            raise NotImplementedError(
                f"Distribution of type {dist_type} is not implemented"
            )

    return dist, sup_title


def plot_and_save(
    dist: BernoulliDistribution
    | CategoricalDistribution
    | ExponentialDistribution
    | LogNormalDistribution
    | NormalDistribution
    | PoissonDistribution
    | TriangularDistribution
    | TruncatedNormalDistribution
    | UniformDistribution,
    sup_title: str,
):
    """Generates a dual-panel plot for a distribution and saves it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    fig: Figure
    ax1: Axes
    ax2: Axes

    # Determine support/range
    if dist.is_continuous:
        # Use PPF to find reasonable bounds (0.1% to 99.9%)
        try:
            x_min, x_max = dist.ppf(0.001), dist.ppf(0.999)
        except Exception:  # Fallback for edge cases
            x_min, x_max = -5, 5
        x = np.linspace(x_min, x_max, 500)
        y_pdf = dist.pdf(x)
        y_cdf = dist.cdf(x)

        # Plot PDF with shading
        ax1.plot(x, y_pdf, lw=2, color=ORANGE)
        ax1.fill_between(x, y_pdf, alpha=0.2, color=ORANGE)
        ax1.set_ylabel("Density (PDF)")

        # Plot CDF with shading
        ax2.plot(x, y_cdf, lw=2, color=BLUE)
        ax2.fill_between(x, y_cdf, alpha=0.2, color=BLUE)
        ax2.set_ylabel("Cumulative Prob (CDF)")

    else:
        # Discrete logic (Bernoulli, Poisson, Categorical)
        if isinstance(dist, CategoricalDistribution):
            x = np.arange(len(dist.choices))
            eval_at = dist.categories
            labels = [str(c) for c in dist.choices]

            ax1.set_xticks(x, labels, rotation=45)
            ax2.set_xticks(x, labels, rotation=45)
        else:
            if isinstance(dist, PoissonDistribution):
                x = np.arange(0, 10)
                eval_at = x
            elif isinstance(dist, BernoulliDistribution):
                x = np.array([0, 1])
                eval_at = x
            else:
                raise NotImplementedError(
                    f"Distribution of {type(dist)=} not supported"
                )
            labels = x

        y_pmf = [dist.pmf(val) for val in eval_at]  # pyright: ignore[reportArgumentType]
        y_cdf = [dist.cdf(val) for val in eval_at]  # pyright: ignore[reportArgumentType]

        # Plot PMF
        ax1.stem(
            x,
            y_pmf,  # pyright: ignore[reportArgumentType]
            basefmt=" ",
            linefmt=ORANGE,
            markerfmt="o",
        )
        ax1.set_ylabel("Probability (PMF)")

        # Plot CDF (Step)
        ax2.step(x, y_cdf, where="post", color=BLUE, lw=2)  # pyright: ignore[reportArgumentType]
        ax2.fill_between(x, y_cdf, step="post", alpha=0.2, color=BLUE)  # pyright: ignore[reportArgumentType]

        # Optional: step plots don't fill_between well, but you can fill_between x and y_cdf
        # with step mode if you really want color under the staircase.
        ax2.set_ylabel("Cumulative Prob (CDF)")

    # Cleanup and labeling

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    fig.suptitle(sup_title, fontsize=14)

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Value")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / f"{dist.dist_type.lower()}.png", dpi=DPI)
    plt.close(fig)


def main():
    for dist_type in DistType:
        print(f"{dist_type=}")
        dist, label = get_dist(dist_type)
        plot_and_save(dist, label)
        # breakpoint()


if __name__ == "__main__":
    main()
