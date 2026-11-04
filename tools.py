
import numpy as np
import matplotlib.pyplot as plt

def rounds(x, sig_figs):
    """
    Rounds a number x to a specified number of significant figures.
    """
    if x == 0:
        return 0.0
    
    # 1. Determine the magnitude (power of 10) of the number
    # np.floor(np.log10(abs(x))) gives the exponent of the largest power of 10 less than |x|.
    magnitude = np.floor(np.log10(np.abs(x)))
    
    # 2. Calculate the decimal place to round to
    # We want to round to (sig_figs - 1) places relative to the magnitude.
    # E.g., if magnitude is 2 (100s) and sig_figs is 3, we round to 2 - 3 + 1 = 0 (ones place).
    decimal_place = sig_figs - 1 - magnitude
    
    # 3. Use numpy.round to perform the rounding
    return np.round(x, int(decimal_place))


def generate_save_figs(fn, pha, grid, peaks=None, c=1):
    grid.plot2d(pha.eval(grid))
    limy = plt.ylim()
    limx = plt.xlim()
    plt.axis('off')
    plt.savefig(f'{fn}.jpg')
    plt.show()

    plt.ylim(limy)
    plt.xlim(limx)
    pha.overlay()
    if peaks is not None:
        plt.plot(peaks['dm_star'], peaks['m_star'], '.')
        # plt.plot(peaks['dm_star'] * c1, peaks['m_star'], '.')
        plt.plot(peaks['dm_star'], c * peaks['m_star'], '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().set_box_aspect(1)
    plt.savefig(f'{fn}_axes.svg')
    plt.show()
