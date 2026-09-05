"""Small local helpers used by the light-curve plotting code."""

import numpy as np


def _validate_bin_size(bin_size):
    if not isinstance(bin_size, (int, np.integer)) or bin_size < 1:
        raise ValueError("bin_size must be a positive integer")


def rebin(values, bin_size):
    """Return the mean of consecutive bins, dropping an incomplete final bin."""
    _validate_bin_size(bin_size)
    values = np.asarray(values)
    n_bins = values.shape[0] // bin_size
    if n_bins == 0:
        return np.array([], dtype=float)
    trimmed = values[:n_bins * bin_size]
    return np.nanmean(trimmed.reshape(n_bins, bin_size), axis=1)


def rebin_std(values, bin_size):
    """Return the sample standard deviation in each consecutive bin."""
    _validate_bin_size(bin_size)
    values = np.asarray(values)
    n_bins = values.shape[0] // bin_size
    if n_bins == 0:
        return np.array([], dtype=float)
    trimmed = values[:n_bins * bin_size]
    return np.nanstd(trimmed.reshape(n_bins, bin_size), axis=1, ddof=1)
