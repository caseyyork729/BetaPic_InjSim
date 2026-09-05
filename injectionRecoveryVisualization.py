import os
import numpy as np
import matplotlib.pyplot as plt
from MULT_DirectPlanetLightCurveAnalysis.misc import rebin
from scipy import stats
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.stats import chi2

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


def delta_bic_flat_vs_sinusoid(t, y, yerr, amp, period):
    """
    Compute ΔBIC = BIC_sinusoid - BIC_flat for a light curve.

    Models:
      flat:      y = c
      sinusoid:  y = c + amp * sin(2π t / period + phi)
    with amp and period fixed; (c, phi) fit by weighted least squares.

    Parameters
    ----------
    t, y, yerr : array_like
        Time, measurement, and 1σ uncertainty.
    amp, period : float
        Fixed sinusoid amplitude and period.

    Returns
    -------
    delta_bic : float
        BIC_sinusoid - BIC_flat (negative favors sinusoid).
    phi_hat : float
        Best-fit phase in [-π, π].
    c_flat : float
        Best-fit constant for flat model.
    c_sin : float
        Best-fit offset for sinusoid model.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    t, y, yerr = t[m], y[m], yerr[m]
    n = y.size
    if n < 3:
        raise ValueError("Need at least 3 valid points.")

    w = 1.0 / (yerr * yerr)
    two_pi_over_p = 2.0 * np.pi / float(period)

    # Flat model: weighted mean
    c_flat = np.sum(w * y) / np.sum(w)
    chi2_flat = np.sum(((y - c_flat) / yerr) ** 2)
    k_flat = 1

    # Sinusoid model: fit (c, phi) with amp, period fixed
    def resid(p):
        a, c, phi = p
        yhat = c + a * np.sin(two_pi_over_p * t + phi)
        return (y - yhat) / yerr

    # Robust-ish initial guess: start from flat offset and phi=0
    p0 = np.array([amp, c_flat, 0.0], dtype=float)
    sol = least_squares(resid, p0, method="trf")
    a_sin, c_sin, phi_hat = sol.x
    phi_hat = (phi_hat + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-π, π]
    chi2_sin = np.sum(sol.fun ** 2)
    k_sin = 4

    # BIC = chi2 + k ln(n) (Gaussian with known σ; constants cancel in ΔBIC)
    bic_flat = chi2_flat + k_flat * np.log(n)
    bic_sin = chi2_sin + k_sin * np.log(n)
    print(chi2_flat, chi2_sin)
    return (bic_sin - bic_flat)

filter_name = 'F410M'
plot_PA = 210
saveFigure = True

chisq_non_variable_amplitudes = np.zeros(5)
chisq_variable_amplitudes = np.zeros(5)
deltaBIC = np.zeros(5)
plt.close('all')
for i, PA in enumerate([90, 150, 210, 270, 330]):
    if filter_name == 'F210M':
        rootDIR_noVar = f'../../Data/{filter_name}_LIKELY_Th8/simulated_data/Sep_540_PA_{PA}_Flux_1450000_VAR0.0_DXY0.0'
        rootDIR_Var = f'../../Data/{filter_name}_LIKELY_Th8/simulated_data/Sep_540_PA_{PA}_Flux_1450000_VAR1.0_DXY0.0'
    if filter_name == 'F410M':
        rootDIR_noVar = f'../../Data/{filter_name}_LIKELY_Th8/simulated_data/Sep_540_PA_{PA}_Flux_420000_VAR0.0_DXY0.0'
        rootDIR_Var = f'../../Data/{filter_name}_LIKELY_Th8/simulated_data/Sep_540_PA_{PA}_Flux_420000_VAR1.0_DXY0.0'

    time0, lc_noVar0 = np.loadtxt(os.path.join(rootDIR_noVar, 'lightcurves', 'Planet_LightCurve_Corrected_Roll_0.txt'), unpack=True)
    time1, lc_noVar1 = np.loadtxt(os.path.join(rootDIR_noVar, 'lightcurves', 'Planet_LightCurve_Corrected_Roll_1.txt'), unpack=True)
    time = np.concatenate([time0, time1])
    lc_noVar = np.concatenate([lc_noVar0, lc_noVar1])
    lc_noise_noVar = np.ones_like(lc_noVar) * estimate_noise_median(lc_noVar)

    time0, lc_Var0 = np.loadtxt(os.path.join(rootDIR_Var, 'lightcurves', 'Planet_LightCurve_Corrected_Roll_0.txt'), unpack=True)
    time1, lc_Var1 = np.loadtxt(os.path.join(rootDIR_Var, 'lightcurves', 'Planet_LightCurve_Corrected_Roll_1.txt'), unpack=True)
    time = np.concatenate([time0, time1])
    t_hr = (time - time[0]) * 24
    t_hr0 = (time0 - time[0]) * 24
    t_hr1 = (time1 - time[0]) * 24
    lc_Var = np.concatenate([lc_Var0, lc_Var1])
    lc_noise_Var = np.ones_like(lc_Var) * estimate_noise_median(lc_Var)

    if PA == plot_PA:
        fig, axes = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'hspace': 0})        
        axes[0].plot(rebin(t_hr0, 20), rebin(lc_noVar0, 20), color='C0', ds='steps-mid')
        axes[0].plot(rebin(t_hr1, 20), rebin(lc_noVar1, 20), color='C0', ds='steps-mid')
        axes[1].plot(rebin(t_hr0, 20), rebin(lc_Var0, 20), color='C1', ds='steps-mid')
        axes[1].plot(rebin(t_hr1, 20), rebin(lc_Var1, 20), color='C1', ds='steps-mid')
        axes[0].axhline(1.0, color='0.5', linestyle='--')
        axes[1].axhline(1.0, color='0.5', linestyle='--')
        axes[1].set_xlabel('Time [hr]')
        fig.text(0.0, 0.5, 'Normalized flux', va='center', rotation='vertical', weight='bold')
        axes[0].set_title(f'Injection/Recovery at PA={PA}° -- {filter_name}', weight='bold')
        axes[0].set_ylim([0.97, 1.03])
        axes[1].set_ylim([0.97, 1.03])
        axes[0].set_ylabel(' ')
        fig.tight_layout()
        if saveFigure:
            plt.savefig(f'../../PaperPlots/Injected_LightCurves_PA_{plot_PA}_{filter_name}.pdf', bbox_inches='tight')

    chisq_noVar, _, _, _ = test_deviation_from_flat(lc_noVar, lc_noise_noVar)
    chisq_non_variable_amplitudes[i] = chisq_noVar

    chisq_variable, _, _, _ = test_deviation_from_flat(lc_Var, lc_noise_Var)
    chisq_variable_amplitudes[i] = chisq_variable
    deltaBIC[i] = delta_bic_flat_vs_sinusoid(t_hr, lc_Var, lc_noise_Var, amp=0.0, period=8.5)

print(deltaBIC)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(np.arange(5)-0.15, chisq_non_variable_amplitudes, width=0.3, color='C0', alpha=0.7, label='Non-variable injection')
ax.bar(np.arange(5)+0.15, chisq_variable_amplitudes, width=0.3, color='C1', alpha=0.7, label='Variable injection')
threshold = chi2.ppf(1 - 2.87e-7, df=len(time) - 1) / (len(time) - 1)
if filter_name == 'F410M':
    ax.legend(loc='upper right', fontsize=12)
ax.axhline(threshold, color='k', linestyle='--')
if filter_name == 'F210M':
    ax.axhline(1.26, color='r', linestyle='--')
    ax.set_ylim([0, 2.0])
if filter_name == 'F410M':
    ax.axhline(1.51, color='r', linestyle='--')
    ax.set_ylim([0, 2.8])
ax.set_xticks(np.arange(5))
ax.set_xticklabels([90, 150, 210, 270, 330])

ax.set_ylabel('Reduced $\chi^2$', weight='bold')
ax.set_xlabel('Injected Position Angle [deg]', weight='bold')
fig.tight_layout()
if saveFigure:
    plt.savefig(f'../../PaperPlots/InjectionRecoverySignificance_PA_{plot_PA}_{filter_name}.pdf', bbox_inches='tight')
