import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.stats import chi2
from pathlib import Path
import psfInjectionConfig_F410M as config

FILTER_NAME = "F410M"

# COPIED FROM MISC.PY BECAUSE IT DIDN'T WANT TO IMPORT
def _validate_bin_size(bin_size):
    if not isinstance(bin_size, (int, np.integer)) or bin_size < 1:
        raise ValueError("bin_size must be a positive integer")

# COPIED FROM MISC.PY BECAUSE IT DIDN'T WANT TO IMPORT
def rebin(values, bin_size):
    """Return the mean of consecutive bins, dropping an incomplete final bin."""
    _validate_bin_size(bin_size)
    values = np.asarray(values)
    n_bins = values.shape[0] // bin_size
    if n_bins == 0:
        return np.array([], dtype=float)
    trimmed = values[:n_bins * bin_size]
    return np.nanmean(trimmed.reshape(n_bins, bin_size), axis=1)

def estimate_noise_median(data, filter_size=50):
    """
    Estimate noise by subtracting median-filtered signal and computing std.
    
    Parameters:
    -----------
    data : array_like
        1D input signal
    filter_size : int, optional
        Size of median filter window (default=50)
    
    Returns:
    --------
    noise_estimate : float
        Standard deviation of residual after median filtering
    residual : array_like
        High-frequency residual (data - median_filtered)
    """
    # Apply median filter to remove low-frequency trends
    median_filtered = median_filter(data, size=filter_size)
    
    # Calculate residual (high-frequency component)
    residual = data - median_filtered
    
    # Estimate noise as standard deviation of residual
    noise_estimate = np.std(residual)
    
    return noise_estimate

def test_deviation_from_flat(values, errors):
    """
    Test 1 & 2: Test if sample deviates from a flat line
    Uses chi-squared test with known uncertainties
    """
    # Weighted mean (best estimate of flat line level)
    weights = 1.0 / errors**2
    weighted_mean = np.average(values, weights=weights)
    
    # Chi-squared statistic
    chi2_stat = np.sum((values - weighted_mean)**2 / errors**2)
    
    # Degrees of freedom
    dof = len(values) - 1
    
    # p-value
    p_value = stats.chi2.sf(chi2_stat, dof)
    
    return chi2_stat / dof, p_value, dof, weighted_mean

filter_name = 'F410M'

chisq_non_variable_amplitudes = np.zeros(5)
chisq_variable_amplitudes = np.zeros(5)
deltaBIC = np.zeros(5)
plt.close('all')

def save_variability_visualization(run_dir, sep, flux, amplitude, pa):
    lightcurve_dir = Path(run_dir) / "lightcurves"

    viz_root = Path(f"{config.DATA_ROOT}/MULT_VizPlots")
    viz_dir = viz_root / f"Sep_{int(sep)}_Flux_{int(flux)}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    output_file = (viz_dir / f"VAR_{amplitude*100:.4f}_PA_{int(pa)}.png")

    # Don't regenerate an existing visualization
    if output_file.exists():
        print()
        print(f"VARIABLE LIGHTCURVE PLOT ALREADY EXISTS: {viz_dir.name}/{output_file.name}")
        return

    # Find lightcurve files for each roll and unpack data
    roll0_file = (lightcurve_dir / "Planet_LightCurve_Corrected_Roll_0.txt")
    roll1_file = (lightcurve_dir / "Planet_LightCurve_Corrected_Roll_1.txt")

    if not roll0_file.exists() or not roll1_file.exists():
        print()
        print("WARNING: Cannot create visualization.")
        print("Missing light curve files:")
        print(f"    {roll0_file}")
        print(f"    {roll1_file}")
        return

    time0, lc0 = np.loadtxt(
        roll0_file,
        unpack=True,
    )

    time1, lc1 = np.loadtxt(
        roll1_file,
        unpack=True,
    )

    time_start = time0[0]
    t_hr0 = (time0 - time_start) * 24
    t_hr1 = (time1 - time_start) * 24

    # Combined light curve for the statistic
    time = np.concatenate([time0, time1])
    lc = np.concatenate([lc0, lc1])

    # Calculate chi square value
    lc_noise = (
        np.ones_like(lc)
        * estimate_noise_median(lc)
    )

    chisq, _, _, _ = test_deviation_from_flat(
        lc,
        lc_noise,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        constrained_layout=True,
    )

    # Rebin and plot
    ax.plot(
        rebin(t_hr0, 20),
        rebin(lc0, 20),
        drawstyle="steps-mid",
        linewidth=1.2,
        label="Roll 0",
    )

    ax.plot(
        rebin(t_hr1, 20),
        rebin(lc1, 20),
        drawstyle="steps-mid",
        linewidth=1.2,
        label="Roll 1",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Normalized flux",
    )

    # Format
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(
        f"Variability Injection Recovery\n"
        f"{FILTER_NAME} | Sep = {sep} mas | "
        f"Flux = {flux} | PA = {pa}°\n"
        f"Amplitude = {amplitude:.6f} | "
        rf"Reduced $\chi^2$ = {chisq:.4f}"
    )

    ax.grid(
        alpha=0.2,
        linewidth=0.7,
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    ymin = np.nanmin(
        np.concatenate([lc0, lc1])
    )
    ymax = np.nanmax(
        np.concatenate([lc0, lc1])
    )

    padding = 0.10 * (ymax - ymin)

    if padding > 0:
        ax.set_ylim(
            ymin - padding,
            ymax + padding,
        )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)
    print()
    print(f"SAVED VARIABLE LIGHTCURVE: {viz_dir.name}/{output_file.name}")

def save_chisq_by_pa_visualization(result):
    # Unpack from results
    sep = result["separation"]
    flux = result["flux"]
    amp_max = float(result["amp_max"])
    pas = result["PAs"]
    threshold = result["threshold"]
    history = result["iteration_history"]

    # Determine which iteration produced limiting amplitude
    matching_iterations = [
        entry
        for entry in history
        if np.isclose(
            round(float(entry["amplitude"]), 6),
            round(amp_max, 6),
            rtol=0.0,
            atol=1e-12,
        )
    ]

    # Check the value from result is in the interation history
    if not matching_iterations:
        print()
        print("=" * 70)
        print("WARNING: COULD NOT FIND amp_max IN ITERATION HISTORY")
        print("=" * 70)
        print(f"    Sep:     {sep}")
        print(f"    Flux:    {flux}")
        print(f"    amp_max: {amp_max:.12f}")
        print()
        print("Available amplitudes:")

        for entry in history:
            print(
                f"    Iteration {entry['iteration']}: "
                f"A={float(entry['amplitude']):.12f}"
            )

        print()
        print("χ² BY PA VISUALIZATION WILL NOT BE GENERATED")
        return

    final_iteration = max(
        matching_iterations,
        key=lambda entry: int(entry["iteration"]))

    # Get amplitude and chi sqaure from the right iteration
    amplitude = float(final_iteration["amplitude"])
    chisq_all = final_iteration["stat_all"]

    # Create plot
    viz_root = Path(f"{config.DATA_ROOT}/MULT_VizPlots")
    viz_dir = viz_root / f"Sep_{int(sep)}_Flux_{int(flux)}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_file = (viz_dir / f"chi2_by_PA_VAR_{amplitude*100:.4f}.png")

    if output_file.exists():
        print(f"χ² BY PA PLOT ALREADY EXISTS: {viz_dir.name}/{output_file.name}")
        return

    pas = np.asarray(pas)
    chisq_all = np.asarray(chisq_all)

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        constrained_layout=True,
    )

    bars = ax.bar(
        pas,
        chisq_all,
        width=0.65 * np.min(np.diff(np.sort(pas)))
        if len(pas) > 1
        else 20,
        alpha=0.8,
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            rf"$5\sigma$ threshold = "
            f"{threshold:.3f}"
        ),
    )

    # Formatting
    ymax = max(
        np.max(chisq_all),
        threshold,
    )

    for bar, chisq in zip(bars, chisq_all):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025 * ymax,
            f"{chisq:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Injected position angle [deg]")
    ax.set_ylabel(r"Reduced $\chi^2$")
    ax.set_title(
        f"Variability Detection by Position Angle\n"
        f"{FILTER_NAME} | Sep = {sep} mas | "
        f"Flux = {flux} | A = {amplitude:.6f}"
    )

    ax.set_xticks(pas)

    ax.set_ylim(
        bottom=0,
        top=ymax * 1.18 if ymax > 0 else 1,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.7,
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)
    print()
    print(f"SAVED χ² BY PA PLOT: {viz_dir.name}/{output_file.name}")