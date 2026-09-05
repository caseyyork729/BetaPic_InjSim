"""
return a model light curve for Beta Pic the star
"""

import numpy as np
from astropy.time import Time
import gc

class BetaPicStellarLC:
    """
    stellar LC fitting results
    F210M
    Period 1 = 0.504900:
    Amplitude = 0.000213
    Phase = 1.434358 rad
    Period 2 = 0.528400:   
    Amplitude = 0.000192, 
    Phase = 2.900897 rad
    Period 3 = 0.483500:  
    Amplitude = 0.000202
    Phase = -1.458178 rad
    Period 4 = 0.442400:
    Amplitude = 0.000235
    Phase = -1.775964 rad

    F410M
    P1 = 0.504900
    Amplitude = 0.000127
    Phase = 0.737854

    Period 2 = 0.528400:  
    Amplitude = 0.000017
    Phase = 1.675902 rad

    Period 3 = 0.483500:  
    Amplitude = 0.000138
    Phase = -1.637056 rad

    Period 4 = 0.442400:  
    Amplitude = 0.000143
    Phase = -1.681347 rad
    """
    def __init__(self, t, t0=60755.475234183694):
        """
        Note the t0 is set to the beginning of the observations

        input time needs to be in BJD (MJD), same as the format in the fits header
        """                
        self.t_BJD = t
        self.t_hours = (t - t0) * 24  # Hours from start
    
        F210M_sine = [{'Period': 0.504900,
                      'Amplitude': 0.000213,
                      'Phase': 1.434358},
                      {'Period': 0.528400, 
                        'Amplitude': 0.000192,
                        'Phase': 2.900897},
                      {'Period': 0.483500,
                       'Amplitude': 0.000202,
                       'Phase': -1.458178},
                       {'Period': 0.442400,  
                        'Amplitude': 0.000235,
                        'Phase': -1.775964}]
        self.F210M_LC = np.ones_like(t)  # will implement it later
        for sineParams in F210M_sine:
            sine_i = sineParams['Amplitude'] * np.sin((self.t_hours / sineParams['Period']) * 2 * np.pi + sineParams['Phase'])
            self.F210M_LC += sine_i
        F410M_sine = [{'Period': 0.504900,
                      'Amplitude': 0.000127,
                      'Phase': 0.737854},
                      {'Period': 0.528400, 
                        'Amplitude': 0.000017,
                        'Phase': 1.675902},
                      {'Period': 0.483500,
                       'Amplitude': 0.000138,
                       'Phase': -1.637056},
                       {'Period': 0.442400,  
                        'Amplitude': 0.000143,
                        'Phase': -1.681347}]
        # calculate the F410M light curve
        self.F410M_LC = np.ones_like(t)
        for sineParams in F410M_sine:
            sine_i = sineParams['Amplitude'] * np.sin((self.t_hours / sineParams['Period']) * 2 * np.pi + sineParams['Phase'])
            self.F410M_LC += sine_i

    def plotLightCurve(self):
        """
        show the light curve
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.t_BJD, self.F210M_LC, label='F210M')
        ax.plot(self.t_BJD, self.F410M_LC, label='F410M')
        ax.set_xlabel('Time [BJD]')
        ax.set_ylabel('Flux')
        ax.set_ylim([0.998, 1.002])
        ax.legend()
        return fig, ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t0=60755.475234183694
    t1=t0 + 1
    t = np.linspace(t0, t1, 1000)
    lc = BetaPicStellarLC(t)
    lc.plotLightCurve()
    plt.show()
    plt.close()
    gc.collect()