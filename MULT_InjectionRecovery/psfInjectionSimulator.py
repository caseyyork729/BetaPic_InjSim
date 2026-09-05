#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for PSF injection simulation.

This script injects simulated PSF images into JWST data to create synthetic
planet signals with time-variable flux modulation.

Modified to inject simulated PSFs at various separations, PAs, fluxes, and 
most importantly, variability amplitudes, to determine the smallest discernable
flux variation amplitude
"""

# Imports
import os
import numpy as np
import time
import shutil
import warnings
import h5py
from pathlib import Path
import re
import pickle
import gc
import pandas as pd
from astropy.io import fits

from scipy import stats
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer

from concurrent.futures import ProcessPoolExecutor
import spaceKLIP

from psfInjectionUtils import process_single_file, create_directories, load_psf_model
from recoverInjectedLightCurve import run_analysis
from psfInjectionVisualization import save_variability_visualization, save_chisq_by_pa_visualization

import tracemalloc
tracemalloc.start()

# Import configuration and utilities
FILTER_NAME = "F410M"
if FILTER_NAME == 'F210M':
    import BetaPic.MULT_InjectionRecovery.psfInjectionConfig_F210M as config
elif FILTER_NAME == 'F410M':
    import BetaPic.MULT_InjectionRecovery.psfInjectionConfig_F410M as config
else:
    raise ValueError("JWST_FILTER must be F210M or F410M")

# Get filepaths from config
PSF_FILENAME = config.PSF_FILENAME
DATA_ROOT = config.DATA_ROOT
INPUT_DATA_DIR = config.INPUT_DATA_DIR
PSF_LIBRARY_DIR = config.PSF_LIBRARY_DIR
SIMDATA_ROOT = config.SIMDATA_ROOT
CHI2_PLOT_DIR = Path(DATA_ROOT) / "MULT_chi2Plots"

def create_processing_config(sep, flux, pa, amp):
    
    def generate_comparison_aperture_positions(
        injection_sep,
        injection_pa,
        n_comparison,
    ):

        if n_comparison < 1:
            return [], []

        n_inner = int(np.ceil(n_comparison / 2))
        n_outer = n_comparison - n_inner

        inner_offsets = []

        if n_inner == 1:
            inner_offsets = [180.0]

        else:
            inner_offsets = np.linspace(
                -135.0,
                135.0,
                n_inner,
            )

        inner_pas = [
            round((injection_pa + offset) % 360.0)
            for offset in inner_offsets
        ]

        inner_seps = [
            injection_sep
            for _ in range(n_inner)
        ]

        outer_pas = []
        outer_seps = []

        if n_outer > 0:
            outer_offsets = np.linspace(
                -90.0,
                180.0,
                n_outer,
                endpoint=False,
            )

            outer_pas = [
                round((injection_pa + offset) % 360.0)
                for offset in outer_offsets
            ]

            outer_sep = round(injection_sep * 1.5)

            outer_seps = [
                outer_sep
                for _ in range(n_outer)
            ]


        comparison_pa_list = (
            inner_pas +
            outer_pas
        )

        comparison_separation_list = (
            inner_seps +
            outer_seps
        )

        assert len(comparison_pa_list) == n_comparison
        assert len(comparison_separation_list) == n_comparison

        # Make sure no comparison aperture at the injection
        for cpa, csep in zip(
            comparison_pa_list,
            comparison_separation_list,
        ):
            if (
                np.isclose(cpa, injection_pa, atol=1e-8)
                and
                np.isclose(csep, injection_sep, atol=1e-8)
            ):
                raise ValueError(
                    "Generated comparison aperture overlaps "
                    "the injection location."
                )

        return (
            comparison_pa_list,
            comparison_separation_list,
        )
    
    # Info from the config file
    period = config.FLUX_PERIOD
    dxy = config.DXY
    dxy_multiplier = config.DXY_MULTIPLIER
    phase = np.random.default_rng(int(flux * pa)).uniform(-np.pi, np.pi)
    n_comparison = config.N_COMPARISON
    
    # Create run label
    run_label = (
        f"Sep_{int(sep)}"
        f"_PA_{int(pa)}"
        f"_Flux_{int(flux)}"
        f"_P_{period:.2f}"
        f"_VAR{amp * 100:.4f}"
        f"_DXY{dxy * dxy_multiplier * 100:.1f}"
        f"_PHASE{phase:.3f}")

    # Filepaths
    output_root = Path(SIMDATA_ROOT) / run_label
    
    # Determine comparison aperture locations
    (comparison_pa_list, 
     comparison_separation_list) = generate_comparison_aperture_positions(sep,
                                                                          pa,
                                                                          n_comparison)

    # Avoid real planet and current injection location
    for i, (cpa, csep) in enumerate(zip(comparison_pa_list, comparison_separation_list)):

        if (np.isclose(csep, config.SEP_BETA_PIC_B, atol=50, rtol=0)
            and
            np.isclose(cpa, config.PA_BETA_PIC_B, atol=10, rtol=0)):
            # Move this comparison aperture by 30 degrees.
            comparison_pa_list[i] = (cpa + 30.0) % 360.0

    return {
        "separation": sep,
        "pa": pa,

        "flux": flux,
        "amplitude": amp*flux,
        "period": period,
        "phase": phase,
        "dxy": dxy,
        "dxy_multiplier": dxy_multiplier,
        
        "pid": 4758,
        "observation_key": config.OBSERVATION_KEY,
        "pixel_scale": config.TARGET_PIXEL_SCALE,
        "star_x": config.STAR_X_CENTER,
        "star_y": config.STAR_Y_CENTER,
        "t0": config.t0,
        "sep_b_pic_b": config.SEP_BETA_PIC_B,
        "pa_b_pic_b": config.PA_BETA_PIC_B,
        "flux_b_pic_b": config.FLUX_BETA_PIC_B,
        "filter": FILTER_NAME,
        "comparison_pa_list": comparison_pa_list,
        "comparison_separation_list": comparison_separation_list,
        "aperture_radius": config.APERTURE_RADIUS,
    
        "output_data_root": str(output_root),
        "output_data_dir": os.path.join(output_root, "aligned"),
        "planet_contribution_dir": os.path.join(output_root, "planet_contribution"),
        "lightcurve_dir": os.path.join(output_root, "lightcurves"),
        "plot_dir": os.path.join(output_root, "plots", "Aperture_Photometry_Results"),

        "label": run_label,
    }    

def setup_directories(processing_config):
    """Create necessary directories for output files."""
    dirs = [processing_config["output_data_dir"], processing_config["planet_contribution_dir"]]
    create_directories(dirs, config.VERBOSE)

def load_and_validate_psf():
    """Load and validate the PSF model."""
    psf_path = os.path.join(PSF_LIBRARY_DIR, PSF_FILENAME)
    
    if config.VERBOSE:
        print(f"Loading PSF model from: {psf_path}")
    
    psf_model = load_psf_model(psf_path)
    
    if config.VERBOSE:
        print(f"PSF model shape: {psf_model.shape}")
        print(f"PSF model peak: {np.max(psf_model):.2e}")
        print(f"PSF model total flux: {np.sum(psf_model):.2e}")
    
    return psf_model

def initialize_database():
    # Create database
    database = spaceKLIP.database.create_database(
        input_dir=INPUT_DATA_DIR,
        file_type='calints.fits',
        output_dir=DATA_ROOT,                                    
        pid=config.PID,
        verbose=config.VERBOSE
    )
    
    # Get key for filter
    key = config.OBSERVATION_KEY
    
    if config.VERBOSE:
        if key in database.obs:
            print(f'Files for {key}:', len(database.obs[key]))
        else:
            print(f"Warning: Key {key} not found in database")
            print("Available keys:", list(database.obs.keys()))
    
    return database, key

def filter_science_files(database, key):
    """Filter to get only science files, excluding reference files."""
    if key not in database.obs:
        raise KeyError(f"Observation key {key} not found in database")
    
    # Get all files
    all_files = database.obs[key]['FITSFILE'].value
    type_vals = database.obs[key]['TYPE'].value
    
    # Filter for science files (exclude reference files)
    # Science files typically have specific pupil values
    science_mask = np.array([type_val == 'SCI' for type_val in type_vals])
    
    science_files = all_files[science_mask]
    
    if config.VERBOSE:
        print(f"Total files: {len(all_files)}")
        print(f"Science files: {len(science_files)}")
        print(f"Reference files: {len(all_files) - len(science_files)}")
    
    return science_files

def prepare_file_list(science_files, processing_config):
    """Prepare list of input files and corresponding output paths."""
    if len(science_files) == 0:
        raise FileNotFoundError("No science files found in database")
    
    print()
    print(f"Found {len(science_files)} science files for injection")
    
    # Create output file paths
    output_files = []
    for input_file in science_files:
        basename = os.path.basename(input_file)
        
        # Add simulation identifier to filename
        name_parts = basename.split('.')
        output_basename = '.'.join(name_parts)
        output_path = os.path.join(processing_config["output_data_dir"], output_basename)
        output_files.append(output_path)
    
    return science_files, output_files
    
def process_files(input_files, output_files, psf_model, processing_config):
    """Process all files with PSF injection."""
    
    # Prepare file processing arguments
    file_args = [
        (input_file, output_file, psf_model, processing_config, processing_config["planet_contribution_dir"])
        for input_file, output_file in zip(input_files, output_files)
    ]
    
    if config.USE_PARALLEL and len(file_args) > 1:
        # Parallel processing
        print(f"Processing {len(file_args)} files in parallel...")
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_file, file_args))

    else:
        # Sequential processing
        print(f"Processing {len(file_args)} files sequentially...")
        
        results = []
        for i, file_arg in enumerate(file_args):
            if config.VERBOSE:
                print(f"Processing file {i+1}/{len(file_args)}: {os.path.basename(file_arg[0])}")
            result = process_single_file(file_arg)
            results.append(result)
            
    del file_args
    gc.collect()
    
    print("Successful!")
    print()
    return results

def print_summary(results, input_files, processing_time, run_config):
    """Print processing summary."""
    successful = sum(1 for r in results if 'Successfully' in r)
    failed = len(results) - successful
    
    # Processing summary
    print(f"\nProcessing Summary:")
    print(f"Total files: {len(input_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {processing_time:.1f} seconds")
    
    # Configuration summary
    print(f"\nInjection Parameters:")
    print(f"Separation: {run_config['separation']:.2f} mas")
    print(f"Position Angle: {run_config['pa']:.2f} degrees")
    print(f"Base Flux: {run_config['flux']:.2f}")
    print(f"Variability Amplitude: {run_config['amplitude']:.2f} "
          f"({100 * run_config['amplitude'] / run_config['flux']:.4f}%)")
    print(f"Flux Period: {run_config['period']:.2f} hours")
    print(f"Phase: {run_config['phase']:.2f} radians")
    print()
    
    # Print any errors
    if failed > 0:
        print(f"\nErrors:")
        for result in results:
            if 'Error' in result:
                print(f"  {result}")

def estimate_noise_median(data, filter_size=50):
    median_filtered = median_filter(data, size=filter_size)
    residual = data - median_filtered
    return np.std(residual)

def test_deviation_from_flat(values, errors):
    weights = 1.0 / errors**2

    weighted_mean = np.average(values, weights=weights)
    chi2_stat = np.sum((values - weighted_mean)**2 / errors**2)
    dof = len(values) - 1

    p_value = stats.chi2.sf(chi2_stat, dof)

    return (
        chi2_stat / dof,
        p_value,
        dof,
        weighted_mean
    )

def validate_inputs():
    """Validate input configuration and directories."""
    
    # Check input directory
    if not os.path.exists(INPUT_DATA_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DATA_DIR}")
    
    # Check PSF file
    psf_path = os.path.join(PSF_LIBRARY_DIR, PSF_FILENAME)
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    
    # Check configuration values
    if np.any(config.TARGET_SEPARATION) <= 0:
        raise ValueError("All TARGET_SEPARATION must be positive")
    
    if config.FLUX_PERIOD <= 0:
        raise ValueError("FLUX_PERIOD must be positive")
    
    if config.FLUX_AMPLITUDE < 0:
        raise ValueError("FLUX_AMPLITUDE must be non-negative")

def set_status(run_dir, status):
    run_dir = Path(run_dir)

    valid_statuses = {"ACTIVE", "SUPERSEDED", "FINALIZED"}
    if status not in valid_statuses:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Expected one of {valid_statuses}."
        )

    # Delete old marker and replace with new
    for old_status in valid_statuses:
        marker = run_dir / old_status

        if marker.exists():
            marker.unlink()

    (run_dir / status).touch()

def get_run_identity(config):
    # All the important values associated with a run
    return {
        "sep": float(config["separation"]),
        "pa": float(config["pa"]),
        "flux": float(config["flux"]),
        "period": float(config["period"]),
        "variability": round((config["amplitude"] / config["flux"]) * 100, 4),
        "dxy": float(config.get("dxy", 0.0)),
    }

def parse_run_label(label):
    # RegEx match the directory name
    pattern = (
        r"^Sep_(?P<sep>[-+0-9.]+)"
        r"_PA_(?P<pa>[-+0-9.]+)"
        r"_Flux_(?P<flux>[-+0-9.]+)"
        r"_P_(?P<period>[-+0-9.]+)"
        r"_VAR(?P<var>[-+0-9.]+)"
        r"_DXY(?P<dxy>[-+0-9.]+)"
        r"_PHASE(?P<phase>[-+0-9.]+)$")

    match = re.match(pattern, label)

    if match is None:
        return None

    return {
        "match": match,
        "sep": float(match.group("sep")),
        "pa": float(match.group("pa")),
        "flux": float(match.group("flux")),
        "period": float(match.group("period")),
        "variability": round(float(match.group("var")), 4),
        "dxy": float(match.group("dxy")),
        "phase": round(float(match.group("phase")), 3),
    }

def same_run_identity(a, b, consider_amp=True, tolerance=1e-8):
    
    # Always consider these
    for key in ("sep", "flux", "pa"):
        if not np.isclose(
            float(a[key]),
            float(b[key]),
            rtol=0,
            atol=tolerance,
        ):
            return False
        
    if consider_amp:
        # Variability is rounded in the directory label
        a_amp = round(float(a["variability"]), 4)
        b_amp = round(float(b["variability"]), 4)

        if not np.isclose(
            a_amp,
            b_amp,
            rtol=0,
            atol=tolerance,
        ):
            return False
    return True

def find_existing_run(config, status=None):

    target = get_run_identity(config)

    # Iterate through all run directories
    for candidate in Path(SIMDATA_ROOT).iterdir():
        if not candidate.is_dir():
            continue
        
        parsed = parse_run_label(candidate.name)
        if parsed is None:
            continue

        if not same_run_identity(target, parsed):
            continue
        
        # Confirm matching status
        if status is not None:
            if not (candidate / status).exists():
                continue

        return candidate

    return None

def supersede_amplitude_runs(sep, flux, amplitude, pas):
    matches_found = False
    
    target_amp = np.round(float(amplitude)*100, 4)
    target_pas = {float(pa) for pa in pas}

    # Iterate through run directories
    for candidate in Path(SIMDATA_ROOT).iterdir():
        if not candidate.is_dir():
            continue
        
        values = parse_run_label(candidate.name)
        if values is None:
            continue
        
        # Iterate through PAs and check if the directory matches
        for pa in target_pas:
            target = {
                "sep": float(sep),
                "flux": float(flux),
                "pa": pa,
                "variability": target_amp,
            }
            
            # Identify runs with the same values
            if not same_run_identity(values, target):
                continue
            matches_found = True

            # Skip runs that are already superseded
            if (candidate / "SUPERSEDED").exists():
                continue
            print(f"    {candidate.name}", end="")

            # Never overwrite FINALIZED
            if (candidate / "FINALIZED").exists():
                print(" -> skipped: FINALIZED")
                continue
            
            # Mark superseded
            set_status(candidate, "SUPERSEDED")
            print(" -> MARKED SUPERSEDED")
    
    # Warning if no matches
    if not matches_found:
        print(
            f"    WARNING: No matching runs found for "
            f"sep={sep}, flux={flux}, "
            f"amplitude={amplitude}")
        
def supersede_runs_outside_bracket(sep, flux, amp_min, amp_max, pas):
    print(
        f"Updating run statuses for current bracket: "
        f"[{amp_min:.6f}, {amp_max:.6f}]")

    current_pas = [float(pa) for pa in pas]

    # Iterate through result dirs
    for candidate in Path(SIMDATA_ROOT).iterdir():
        if not candidate.is_dir():
            continue

        values = parse_run_label(candidate.name)
        if values is None:
            continue
        
        candidate_matches_family = False
    
        # Iterate through PAs
        for pa in current_pas:
            target = {
                "sep": float(sep),
                "flux": float(flux),
                "pa": pa,
                "variability": 0.0,  # ignored
            }

            # Identify matches
            if same_run_identity(values,
                                 target,
                                 consider_amp=False
            ):
                candidate_matches_family = True
                break

        if not candidate_matches_family:
            continue
        
        # Check amplitude separately
        candidate_amp = float(values["variability"]) / 100
        
        if np.isclose(round(candidate_amp, 6), round(amp_min, 6), atol=1e-12):
            continue

        if np.isclose(round(candidate_amp, 6), round(amp_max, 6), atol=1e-12):
            continue

        # Call supersede function for old amplitudes
        if (candidate_amp < amp_min or candidate_amp > amp_max):
            supersede_amplitude_runs(
                sep=sep,
                flux=flux,
                amplitude=candidate_amp,
                pas=pas)
        
def cleanup(after_process, after_analysis):
        
    print()
    print("-" * 70)
    print("STARTING MATPLOTLIB CLEANUP")
    print("-" * 70)
    plt.close("all")
                
    if config.VERBOSE:

        # Line2D objs
        lines = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, matplotlib.lines.Line2D)]

        print("Line2D objects before cleanup:", len(lines))

        for line in lines:
            try:
                line.remove()
            except Exception:
                pass

        del lines

        # ErrorBarContainer objs
        containers = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, ErrorbarContainer)]

        print("ErrorbarContainers before cleanup:", len(containers))

        for container in containers:
            try:
                container.remove()
            except Exception:
                pass

        del containers

        # Axes objs
        axes = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, matplotlib.axes.Axes)]

        print("Axes before cleanup:", len(axes))

        for ax in axes:
            try:
                ax.clear()
            except Exception:
                pass

        del axes

        # Figure objs
        figures = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, matplotlib.figure.Figure)]

        print("Figures before cleanup:", len(figures))

        for fig in figures:
            try:
                fig.clear()
            except Exception:
                pass

        del figures

    # Final collection
    collected = gc.collect()
    print()
    print("GC collected:", collected)

    remaining_lines = sum(
        isinstance(obj, matplotlib.lines.Line2D)
        for obj in gc.get_objects())

    remaining_containers = sum(
        isinstance(obj, ErrorbarContainer)
        for obj in gc.get_objects())
                
    print("Remaining Figure:", len(plt.get_fignums()))
    print("Remaining Line2D:", remaining_lines)
    print("Remaining ErrorbarContainers:", remaining_containers)
                
    print("-" * 70)
    print("MATPLOTLIB CLEANUP COMPLETE")
    print("-" * 70)

    gc.collect()
                
    if config.VERBOSE:
                    
        # Verify complete cleanup
        after_cleanup = tracemalloc.take_snapshot()
                    
        # Compare
        print()
        print("MEMORY GROWTH FROM ANALYSIS")
        print("=" * 70)

        for stat in after_analysis.compare_to(
            after_process,
            "lineno")[:20]:

            print(stat)
                    
        print()
        print("MEMORY RECOVERY FROM CLEANUP")
        print("=" * 70)

        for stat in after_cleanup.compare_to(
            after_analysis,
            "lineno")[:20]:
            
            print(stat)                

        print()
        print("FINAL CLEANUP")
        print("=" * 70)
                    
        for stat in after_cleanup.statistics("lineno")[:20]:
            print(stat)
                    
    plt.close("all")
    gc.collect()

def find_limiting_variability(
    science_files,
    start_time,
    threshold,
    completed_runs,
    sep,
    flux,
    all_pas,
    psf_model,
    amp_tolerance,
    max_iterations=30):
    """
    Find the variability amplitude where the detection
    statistic crosses the specified threshold.

    Parameters
    ----------
    science_files : list
        Science files used for injection

    start_time : float
        Start time of the overall run

    threshold : float
        Detection threshold

    completed_runs : int
        Number of previously completed combinations

    sep : float
        Separation in mas

    flux : float
        Base injection flux

    all_pas : list
        Position angles to evaluate

    psf_model : ndarray
        PSF model

    amp_tolerance : float
        Stop when the amplitude bracket is narrower than this value

    max_iterations : int
        Maximum number of bisection iterations

    Returns
    -------
    dict
        Results from the variability search
    """
    
    # Select PAs
    if (sep == 540) and (30 in all_pas):
        pas = [pa for pa in all_pas if pa != 30]
    else:
        pas = list(all_pas)
    
    def evaluate_amplitude(amplitude, iteration_number, pas):
        """
        Evaluate one variability amplitude across all PAs.
        """
        
        # Initialize
        chisq_all = np.zeros(len(pas))
        noise_all = np.zeros(len(pas))
        std_all = np.zeros(len(pas))

        snapshot_after_process = None
        snapshot_after_analysis = None

        amplitude_start = time.time()

        print()
        print(f"Evaluating amplitude A={amplitude:.6f}")
        print("-" * 70)

        # Loop through PAs
        for i, pa in enumerate(pas):
            start = time.time()
            
            # Create processing config
            run_config = create_processing_config(
                sep,
                flux,
                pa,
                amplitude)
            setup_directories(run_config)
            
            # Find any active run
            existing_active = find_existing_run(
                run_config,
                status="ACTIVE")

            if existing_active is not None:
                print()
                print(f"Found existing ACTIVE run for A={amplitude:.6f}, PA={pa}:")
                print(f"    {existing_active.name}")

                # Get filepath info from existing run
                run_config["output_data_root"] = str(existing_active)
                run_config["label"] = existing_active.name
                run_dir = existing_active

            else:
                # Start a new amplitude run
                run_dir = Path(run_config["output_data_root"])

                set_status(run_dir, "ACTIVE")

                print()
                print(f"Starting new ACTIVE run:")
                print(f"    {run_dir.name}")
            
            result = None
            analysis_result = None
            input_files = None
            output_files = None
            
            # Check for post injection analysis
            completion_file = (run_dir / "analysis_complete.txt")
            if completion_file.exists():
                print("    Post-injection analysis already complete; reusing existing results")

            else:
                print("    Post-injection analysis incomplete; processing now")
                input_files, output_files = (prepare_file_list(science_files, run_config))
                print()
                result = process_files(
                    input_files,
                    output_files,
                    psf_model,
                    run_config,
                )

                # Snapshot
                snapshot_after_process = tracemalloc.take_snapshot()

                # Run analysis
                print("=" * 70)
                print("STARTING RUN_ANALYSIS")
                print("=" * 70)
                
                analysis_result = run_analysis(
                    run_config
                )
                snapshot_after_analysis = tracemalloc.take_snapshot()
                
                # Summarize
                elapsed = time.time() - start
                print_summary(
                    result,
                    input_files,
                    elapsed,
                    run_config)

            # Load lightcurve for each roll
            lightcurve_dir = (run_dir / "lightcurves")
            roll0_file = (lightcurve_dir / "Planet_LightCurve_Corrected_Roll_0.txt")
            roll1_file = (lightcurve_dir / "Planet_LightCurve_Corrected_Roll_1.txt")

            if not roll0_file.exists():
                raise FileNotFoundError(
                    f"Missing Roll 0 light curve:\n"
                    f"{roll0_file}")

            if not roll1_file.exists():
                raise FileNotFoundError(
                    f"Missing Roll 1 light curve:\n"
                    f"{roll1_file}")

            _, lc0 = np.loadtxt(roll0_file, unpack=True)
            _, lc1 = np.loadtxt(roll1_file, unpack=True)
            lc = np.concatenate([lc0, lc1])

            # Calculate chi square
            lc_noise = (np.ones_like(lc) * estimate_noise_median(lc))

            chisq, _, _, _ = (test_deviation_from_flat(lc, lc_noise))
            chisq_all[i] = chisq
            noise_all[i] = estimate_noise_median(lc)
            std_all[i] = np.std(lc)
            
            # Save injected variability lightcurve
            save_variability_visualization(run_config["output_data_root"], 
                                           sep, flux, amplitude, pa)
            print(
                f"A={amplitude:.6f} - "
                f"Completed angle "
                f"{i + 1}/{len(pas)} "
                f"(PA={pa}):"
                f"\n    χ²={chisq:.4f}"
            )

            plt.close('all')
            del input_files, output_files
            del lc0, lc1, lc, lc_noise
            del result, analysis_result
            gc.collect()

        # Average over all PAs
        stat = np.mean(chisq_all)
        noise = np.mean(noise_all)
        std = np.mean(std_all)
        
        # Cleanup if not skipped/repeated
        if snapshot_after_process and snapshot_after_analysis:
            cleanup(snapshot_after_process, snapshot_after_analysis)

        # Append to iteration history
        elapsed = (time.time() - amplitude_start)
        iteration_history.append({
            "iteration": iteration_number,
            "amplitude": amplitude,
            "stat": stat,
            "noise": noise,
            "lc_std": std,
            "stat_all": chisq_all,
            "noise_all": noise_all,
            "lc_std_all": std_all})
        
        # Summarize
        print()
        print(f"Amplitude evaluation complete:")
        print(f"    A = {amplitude:.6f}")
        print(f"    mean reduced χ² = {stat:.6f}")
        for i, pa in enumerate(pas):
            print(f"        PA={pa}: {chisq_all[i]}")
        print(f"    threshold = {threshold:.6f}")
        print(f"    elapsed = {elapsed:.2f} s")

        del chisq_all, noise_all, std_all
        gc.collect()

        return stat
    
    def is_finalized(sep, flux, pa, iteration_history):

        # Get all evaluated amplitudes
        checkpoint_amplitudes = {float(entry["amplitude"]) for entry in iteration_history}

        # Iterate through run directories
        for candidate in Path(SIMDATA_ROOT).iterdir():
            if not candidate.is_dir():
                continue

            # Check for combination/PA match
            if not (f"Sep_{int(sep)}_" in candidate.name
                and f"_PA_{int(pa)}_" in candidate.name
                and f"_Flux_{int(flux)}_" in candidate.name
            ):
                continue

            # Check for finalized status
            if not (candidate / "FINALIZED").exists():
                continue

            # Match amplitude
            match = re.search(
                r"_VAR(?P<amp>[-+0-9.eE]+)_",
                candidate.name)
            if match is None:
                print(f"WARNING: FINALIZED run has no readable amplitude:")
                print(f"    {candidate.name}")
                continue

            # Compare to checkpoints
            finalized_amplitude = float(match.group("amp"))
            amplitude_match = any(
                np.isclose(
                    finalized_amplitude/100,
                    round(checkpoint_amplitude, 6),
                    rtol=0.0,
                    atol=1e-12,
                )
                for checkpoint_amplitude
                in checkpoint_amplitudes
            )

            # Return True if finalized, False otherwise
            if amplitude_match:
                for a in checkpoint_amplitudes:
                    if np.isclose(finalized_amplitude/100, round(a, 6), rtol=0.0, atol=1e-12):
                        match_val = a
                        
                print(f"Confirmed FINALIZED run for PA={pa}:")
                print(f"    Amplitude: {match_val:.6f}")
                print(f"    Directory: {candidate.name}")
                return True

            else:
                print(f"WARNING: FINALIZED run found for "
                      f"PA={pa}, but its amplitude is NOT "
                      f"in the checkpoint")
                print(f"    Directory: {candidate.name}")
                print(f"    Finalized amplitude: {finalized_amplitude:.12f}")
        return False
    
    def save_checkpoint(
        checkpoint_file,
        iteration_history,
        amp_min,
        stat_min,
        amp_max,
        stat_max,
        threshold,
        sep,
        flux,
        pas):
        
        print()
        pa_finalized = np.zeros(len(pas), dtype=bool)
        
        # Iterate through PAs
        for i, pa in enumerate(pas):
            pa_finalized[i] = is_finalized(sep, flux, pa, iteration_history)
            if not pa_finalized[i]:
                print(f"PA = {pa} is not FINALIZED")
        
        # Save finalized checkpoint if all PAs finished
        if np.all(pa_finalized):
            print()
            print("-" * 70)
            print(f"Saving FINALIZED checkpoint: {checkpoint_file.name}")
            print("-" * 70)
            print()
            status = "FINALIZED"
            
        # Save active checkpoint if not all PAs finished
        else:
            print()
            print("-" * 70)
            print(f"Saving ACTIVE checkpoint: {checkpoint_file.name}")
            print("-" * 70)
            status = "ACTIVE"

        # Dump checkpoint to pickle
        checkpoint = {
            "iteration_history": iteration_history.copy(),

            "amp_min": amp_min,
            "stat_min": stat_min,

            "amp_max": amp_max,
            "stat_max": stat_max,

            "threshold": threshold,
            "separation": sep,
            "flux": flux,
            "PAs": pas,

            "status": status,
        }

        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)

    # START OF FUNCTION
    print()
    print("=" * 70)
    print(f"TESTING COMBINATION: sep={sep}, flux={flux}")
    
    # Check for an existing checkpoint
    checkpoint_status = None
    checkpoint_dir = Path(
        f"../../Data/{FILTER_NAME}_LIKELY_Th8/variability_checkpoints"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = (
        checkpoint_dir
        / f"Sep_{sep}_Flux_{flux}_variability_search.pkl"
    )

    active_checkpoint = None
    finalized_checkpoint = None
    iteration_history = []

    # Continue from existing checkpoint
    if checkpoint_file.exists():

        print("CONTINUING FROM CHECKPOINT")
        print("=" * 70)

        checkpoint = pd.read_pickle(checkpoint_file)

        checkpoint_status = checkpoint.get("status", None)
        checkpoint_sep = checkpoint.get("separation", None)
        checkpoint_flux = checkpoint.get("flux", None)
        checkpoint_pas = checkpoint.get("PAs", None)
        
        same_combination = (
            checkpoint_sep == sep
            and checkpoint_flux == flux
            and checkpoint_pas == all_pas)

        print(f"Checkpoint: {checkpoint_file.name}")
        print(f"Status:     {checkpoint_status}")
        print(f"Bracket:    [{checkpoint.get('amp_min'):.6f}, {checkpoint.get('amp_max'):.6f}]")
        print(f"PAs:        {checkpoint_pas}")

        if same_combination:

            # Skip to end if checkpoint is finalized
            if checkpoint_status == "FINALIZED":
                finalized_checkpoint = checkpoint
                print()
                print("-" * 70)
                print("FINALIZED CHECKPOINT FOUND")
                print("SKIPPING COMPLETED AMPLITUDE SEARCH")
                print("-" * 70)
            
            # Continue if checkpoint is active
            elif checkpoint_status == "ACTIVE":
                active_checkpoint = checkpoint
                print()
                print("-" * 70)
                print("ACTIVE CHECKPOINT FOUND")
                print("RESUMING ACTIVE AMPLITUDE SEARCH")
                print("-" * 70)

            else:
                print()
                print(f"WARNING: Unknown checkpoint status: {checkpoint_status!r}")

    # Start new run if no checkpoint
    else:
        print("NO CHECKPOINT FOUND - STARTING NEW RUN")
        print("=" * 70)
    
    # Return finalized run
    if finalized_checkpoint is not None:
        return finalized_checkpoint

    # Continue active run
    elif active_checkpoint is not None:
        iteration_history = active_checkpoint["iteration_history"]

        amp_min = float(active_checkpoint["amp_min"])
        stat_min = float(active_checkpoint["stat_min"])
        amp_max = float(active_checkpoint["amp_max"])
        stat_max = float(active_checkpoint["stat_max"])
        threshold = float(active_checkpoint["threshold"])

        # Get number of previous iterations
        if iteration_history:
            iternum = max(int(entry["iteration"]) for entry in iteration_history) + 1

        else:
            iternum = 1

        print(f"Last completed iteration: {iternum - 1}")
        print(f"Current minimum amplitude: {amp_min:.12f}")
        print(f"Current minimum statistic: {stat_min:.12f}")
        print(f"Current maximum amplitude: {amp_max:.12f}")
        print(f"Current maximum statistic: {stat_max:.12f}")
        print(f"Detection threshold: {threshold:.12f}")

    else:
        print()
        print("STARTING NEW AMPLITUDE SEARCH")
        
        iternum = 1
        
        # Set ref values to estimate limiting amplitude - BASED ON MY OLD RUNS (deleted)
        ref_amp_max = config.FLUX_AMPLITUDE
        ref_flux = max(config.FLUX_BASE)
        ref_sep = min(config.TARGET_SEPARATION)
        
        # Set appropriate min and max 
        amp_min = 0
        amp_max = (
            ref_amp_max
            * (ref_flux / flux) ** 0.8
            * (ref_sep / sep) ** 0.6
        )

        # Evaluate min amplitude
        stat_min = evaluate_amplitude(amp_min, iternum, pas)

        print()
        print(f"Combination {completed_runs + 1}:")
        print(f"    Iteration {iternum} - minimum amplitude:")
        print(f"        A = {amp_min:.6f}")
        print(f"        χ² = {stat_min:.6f}")
        print(f"        threshold = {threshold:.6f}")
        print()

        iternum += 1

        # Evaluate max amplitude
        stat_max = evaluate_amplitude(amp_max, iternum, pas)

        print()
        print(f"Combination {completed_runs + 1}:")
        print(f"    Iteration {iternum} - maximum amplitude:")
        print(f"        A = {amp_max:.6f}")
        print(f"        χ² = {stat_max:.6f}")
        print(f"        threshold = {threshold:.6f}")

        iternum += 1

    assert amp_min < amp_max
    assert stat_min < threshold

    # Expand bracket if necessary
    while stat_max < threshold:
        print()
        print("-" * 70)
        print("UPPER AMPLITUDE BELOW THRESHOLD")
        print("-" * 70)
        print(f"Current upper amplitude: A={amp_max:.6f}")
        print(f"Current statistic:       {stat_max:.6f}")
        print(f"Threshold:               {threshold:.6f}")
        print()
        print("EXPANDING UPPER AMPLITUDE")

        amp_min = amp_max
        stat_min = stat_max
        amp_max = 2.0 * amp_max

        print()
        print(f"New bracket:")
        print(f"    lower: A={amp_min:.6f}")
        print(f"    upper: A={amp_max:.6f}")
        print()

        # Evaluate new upper bound
        stat_max = evaluate_amplitude(amp_max, iternum, pas)
        iternum += 1

        # Safety check to avoid too many iterations
        if iternum > max_iterations:
            raise ValueError(
                "Maximum number of amplitude iterations reached "
                "while expanding the upper amplitude. "
                f"Current A={amp_max:.6f}, "
                f"χ²={stat_max:.6f}, "
                f"threshold={threshold:.6f}")

    # Assert threshold successfully bracketed
    assert amp_min < amp_max
    assert stat_min < threshold <= stat_max

    print("-" * 70)
    print("THRESHOLD SUCCESSFULLY BRACKETED")
    print("-" * 70)
    print(f"lower: A={amp_min:.6f}, χ²={stat_min:.6f}")
    print(f"upper: A={amp_max:.6f}, χ²={stat_max:.6f}")
    print()
        
    # Supersede all old bracketing amplitudes
    supersede_runs_outside_bracket(sep, flux, amp_min, amp_max, pas)
    print()

    # Bisect the bracket until limiting variability is known within the tolerance
    if checkpoint_status != "FINALIZED":
        print("=" * 70)
        print("STARTING BISECTION")
        print("=" * 70)
        for iteration in range(iternum-1, max_iterations):

            # Check if range has been reduced to target
            if (amp_max - amp_min <= amp_tolerance):
                print()
                print("-" * 70)
                print("AMPLITUDE TOLERANCE REACHED")
                print("-" * 70)
                
                # Save checkpoint one last time
                save_checkpoint(
                    checkpoint_file,
                    iteration_history=iteration_history,
                    amp_min=amp_min,
                    stat_min=stat_min,
                    amp_max=amp_max,
                    stat_max=stat_max,
                    threshold=threshold,
                    sep=sep,
                    flux=flux,
                    pas=pas)
                break

            # Take midpoint and evaluate chi square
            amp_mid = (0.5 * (amp_min + amp_max))
            stat_mid = evaluate_amplitude(amp_mid, iteration+1, pas)

            # Output summary
            print()
            print(f"Combination {completed_runs + 1}:")
            print(f"    Iteration {iteration + 1}:")
            print(f"        A = {amp_mid:.6f}")
            print(f"        χ² = {stat_mid:.6f}")
            print(f"        threshold = {threshold:.6f}")

            # Compare stat to threshold
            if stat_mid >= threshold:
                # Sucessful detection
                amp_max = amp_mid
                stat_max = stat_mid

            else:
                # No detection
                amp_min = amp_mid
                stat_min = stat_mid

            # Supersede all old runs
            supersede_runs_outside_bracket(
                sep=sep,
                flux=flux,
                amp_min=amp_min,
                amp_max=amp_max,
                pas=pas,
            )

            # Save ACTIVE checkpoint
            save_checkpoint(
                checkpoint_file,
                iteration_history=iteration_history,
                amp_min=amp_min,
                stat_min=stat_min,
                amp_max=amp_max,
                stat_max=stat_max,
                threshold=threshold,
                sep=sep,
                flux=flux,
                pas=pas)

    # Final result
    total_elapsed = (time.time() - start_time)

    print("-" * 70)
    print("VARIABILITY SEARCH COMPLETED")
    print("-" * 70)

    print(f"Limiting amplitude = {amp_max:.6f}")
    print(f"Final bracket = [{amp_min:.6f}, {amp_max:.6f}]")
    print(f"Final statistic = {stat_max:.6f}")
    print(f"Threshold = {threshold:.6f}")
    print(f"Total elapsed time = {total_elapsed:.2f} s")

    print()
    print("=" * 70)
    print("FINALIZING RUN")
    print("=" * 70)

    # Iterate through PAs
    for pa in pas:
        final_template = create_processing_config(
            sep,
            flux,
            pa,
            amp_max)
        
        # Find existing FINALIZED run
        finalized = find_existing_run(
            final_template,
            status="FINALIZED")
        
        if finalized is not None:
            print()
            print("Found FINALIZED run:")
            print(f"    {finalized.name}")
        
        # If no FINALIZED, find ACTIVE run and finalize it
        else:
            final_active = find_existing_run(
                final_template,
                status="ACTIVE")

            if final_active is not None:
                print()
                print("Finalizing limiting-amplitude run:")
                print(f"    {final_active.name}")

                set_status(final_active, "FINALIZED")

            else:
                print()
                print("WARNING: Could not find ACTIVE run for final amplitude")
        
    # Create final config with flag PA = 999
    final_config = create_processing_config(
        sep,
        flux,
        999,
        amp_max)

    result = {
        "amp_limit": amp_max,
        "stat": stat_max,
        "threshold": threshold,

        "amp_min": amp_min,
        "stat_min": stat_min,
        "amp_max": amp_max,
        "stat_max": stat_max,

        "separation": sep,
        "flux": flux,
        "PAs": pas,

        "config": final_config,
        "iterations": len(iteration_history),
        "iteration_history": iteration_history,
        "elapsed": total_elapsed,
    }
    return result

def plot_iterations(result):
    # Unpack values from result
    sep = result["separation"]
    flux = result["flux"]
    iteration_history = result["iteration_history"]
    threshold = result["threshold"]
    amp_min = result["amp_min"]
    amp_max = result["amp_max"]
    
    # Plot chi2 statistic vs iteration
    chi2_iteration_path = CHI2_PLOT_DIR / f"chi2_vs_iteration_Sep_{int(sep)}_Flux_{int(flux)}.png"
    print()

    if chi2_iteration_path.exists():
        print(f"χ² VS ITERATION PLOT ALREADY EXISTS: {chi2_iteration_path.name}")

    else:
        print("PLOTTING χ² STATISTIC VS ITERATION")

        iterations = np.array([x["iteration"] for x in iteration_history])
        stats = np.array([x["stat"] for x in iteration_history])
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot
        ax.plot(
            iterations,
            stats,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=r"PA-averaged reduced $\chi^2$")

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=rf"5$\sigma$ threshold = {threshold:.3f}")

        ax.scatter(
            iterations[-1],
            stats[-1],
            s=60,
            zorder=5,
            label=rf"Final: $\chi^2_\nu$ = {stats[-1]:.3f}")

        # Formatting
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(r"Reduced $\chi^2$", fontsize=12)
        ax.set_title(f"Variability Search: Separation = {sep} mas, Flux = {flux:g}",
                     fontsize=13,
                     pad=10)

        ax.set_xticks(iterations)

        y_min = min(stats.min(), threshold)
        y_max = max(stats.max(), threshold)

        if np.isclose(y_min, y_max):
            y_min -= 0.1
            y_max += 0.1

        padding = 0.08 * (y_max - y_min)

        ax.set_ylim(
            max(0, y_min - padding),
            y_max + padding)

        ax.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=0.8,
            alpha=0.6)

        ax.legend(
            loc="best",
            frameon=True)

        fig.tight_layout()

        chi2_iteration_path.parent.mkdir(
            parents=True,
            exist_ok=True)

        fig.savefig(
            chi2_iteration_path,
            dpi=200,
            bbox_inches="tight")

        plt.close(fig)
        print(f"SAVED ITERATION PLOT: {chi2_iteration_path}")

    # Plot chi2 statistic vs amplitude
    chi2_amplitude_path = CHI2_PLOT_DIR / f"chi2_vs_amplitude_Sep_{int(sep)}_Flux_{int(flux)}.png"

    if chi2_amplitude_path.exists():
        print(f"χ² VS AMPLITUDE PLOT ALREADY EXISTS: {chi2_amplitude_path.name}")

    else:
        print("PLOTTING χ² STATISTIC VS AMPLITUDE")

        # Get values from iteration history
        amplitudes = np.array([x["amplitude"] for x in iteration_history])
        stats = np.array([x["stat"] for x in iteration_history])

        order = np.argsort(amplitudes)
        amplitudes = np.flip(amplitudes[order])
        stats = np.flip(stats[order])

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot
        ax.plot(
            amplitudes,
            stats,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=r"PA-averaged reduced $\chi^2$")

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=rf"5$\sigma$ threshold = {threshold:.3f}")

        ax.axvline(
            amp_max,
            linestyle=":",
            linewidth=1.5,
            label=f"Limiting amplitude = {amp_max:.6f}")

        # Formatting
        ax.axvspan(
            amp_min,
            amp_max,
            alpha=0.15,
            label=(f"Final bracket = [{amp_min:.6f}, {amp_max:.6f}]"))

        ax.set_xlabel("Variability amplitude", fontsize=12)
        ax.set_ylabel(r"Reduced $\chi^2$", fontsize=12)
        ax.set_title(f"Variability Search: Separation = {sep} mas, Flux = {flux:g}",
                     fontsize=13,
                     pad=10)

        ax.ticklabel_format(
            axis="x",
            style="plain",
            useOffset=False)

        x_min = min(amplitudes.min(), amp_min)
        x_max = max(amplitudes.max(), amp_max)

        if np.isclose(x_min, x_max):
            x_min -= 1
            x_max += 1

        x_padding = 0.05 * (x_max - x_min)

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        y_min = min(stats.min(), threshold)
        y_max = max(stats.max(), threshold)

        if np.isclose(y_min, y_max):
            y_min -= 0.1
            y_max += 0.1

        y_padding = 0.08 * (y_max - y_min)

        ax.set_ylim(
            max(0, y_min - y_padding),
            y_max + y_padding)

        ax.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=0.8,
            alpha=0.6)

        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        chi2_amplitude_path.parent.mkdir(
            parents=True,
            exist_ok=True)

        fig.savefig(
            chi2_amplitude_path,
            dpi=200,
            bbox_inches="tight")

        plt.close(fig)
        print(f"SAVED AMPLITUDE PLOT: {chi2_amplitude_path}")

def delete_superseded(result, delete):
    # Unpack from results
    amp_min = round(float(result["amp_min"])*100, 4)
    amp_max = round(float(result["amp_max"])*100, 4)
    sep = result["separation"]
    flux = result["flux"]
    pas = result["PAs"]
    
    if not delete:
        print("=" * 70)
        print("SKIPPING DELETION OF OLD SUPERSEDED RUNS")
        print("=" * 70)
        return
        
    # Clean superseded runs
    print()
    print("=" * 70)
    print("CLEANING SUPERSEDED RUNS")
    print("=" * 70)

    print()
    print("Final bracket amplitudes:")
    print(f"    {amp_min:.4f}")
    print(f"    {amp_max:.4f}")
    print()
    
    current_pas = [float(pa) for pa in pas]
    
    # Get and parse directories
    for directory in Path(SIMDATA_ROOT).iterdir():
        if not directory.is_dir():
            continue
        
        values = parse_run_label(directory.name)
        if values is None:
            continue
        directory_amplitude = float(values["variability"])

        same_family = False

        for pa in current_pas:
            target = {
                "sep": float(sep),
                "flux": float(flux),
                "pa": pa,
                "variability": 0.0, # ignored
            }

            if same_run_identity(
                values,
                target,
                consider_amp=False,
            ):
                same_family = True
                break

        if not same_family:
            continue
        
        # Lower bound
        if np.isclose(
            directory_amplitude,
            amp_min,
            rtol=0,
            atol=1e-8,
        ):
            print(f"Keeping lower: {directory.name}")
            continue

        # Upper bound
        if np.isclose(
            directory_amplitude,
            amp_max,
            rtol=0,
            atol=1e-8,
        ):
            print(f"Keeping upper: {directory.name}")
            continue
            
        # SUPERSEDED flag
        if not (directory / "SUPERSEDED").exists():
            continue

        print(f"DELETING:    {directory.name}")
        shutil.rmtree(directory)

    print()
    print("-" * 70)
    print("SUPERSEDED DIRECTORY CLEANUP COMPLETE")
    print("-" * 70)

def main():
    print("Starting PSF injection simulation...")
    print()
    
    # Validate inputs
    validate_inputs()
    
    # Initialize spaceKLIP database
    database, key = initialize_database()
    
    # Filter for science files only
    science_files = filter_science_files(database, key)
    
    # Load PSF model
    psf_model = load_and_validate_psf()
    
    # Load data from config
    def to_list(x):
        return [x] if not isinstance(x, list) else list(x)
    
    seps = to_list(config.TARGET_SEPARATION)
    fluxes = to_list(config.FLUX_BASE)
    pas = to_list(config.TARGET_PA)
    
    total = len(seps)*len(fluxes)
    
    complete = 0
    tol = 1e-4
    
    print(f"RUN OVERVIEW - {FILTER_NAME}")
    print("=" * 70)
    print(f"{total} Grid Points")
    print(f"    {len(pas)} Position Angles")
    print(f"        {int(np.ceil(np.log2(config.FLUX_AMPLITUDE / tol)))} "
          f"Amplitude Iterations (minimum)")
    
    # Determine chi2 threshold based on amount of data
    saveFileName = f'../../Data/{FILTER_NAME}_LIKELY_Th8/lightcurves/Host_Lightcurves_MP.h5'
    with h5py.File(saveFileName, 'r') as hf:
        t_BJD = hf['t_BJD'][:]
    t_hours = (t_BJD - t_BJD[0]) * 24
    threshold = stats.chi2.ppf(1 - 2.87e-7, df=len(t_hours) - 1) / (len(t_hours) - 1)

    # Loop through fluxes and separations for grid points
    for flux in fluxes:
        for sep in seps:
            # Process files
            start_time = time.time()
            result = find_limiting_variability(science_files, start_time, threshold, complete,
                                               sep, flux, pas, psf_model, tol)
            
            # Save visualization plots
            plot_iterations(result)
            save_chisq_by_pa_visualization(result)
            
            # Delete old superseded amplitudes
            delete_superseded(result, config.DELETE_SUPERSEDED)
            
            # Cleanup
            del result
            plt.close("all")
            gc.collect()

            complete += 1
            print()
            print(f"PSF INJECTION SIMULATION COMPLETE! ({complete}/{total})")
            print(f"Output files saved in: {SIMDATA_ROOT}")
            
if __name__ == "__main__":
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()