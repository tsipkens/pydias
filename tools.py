
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# ANSI color codes
GREEN = "\033[92m"
BLUE = "\033[96m"
GRAY = "\033[30m"  # alt. 90m
RESET = "\033[0m"

class tqdm2(tqdm):
    def format_meter(self, n, total, elapsed, **kwargs):
        if total:
            frac = n / total
            bar_length = 25

            filled_len = int(bar_length * frac)
            empty_len = bar_length - filled_len

            if frac >= 1:
                bar = (
                    GREEN + "█" * filled_len + RESET +
                    GRAY + "█" * empty_len + RESET
                )
            else:
                bar = (
                    BLUE + "█" * filled_len + RESET +
                    GRAY + "█" * empty_len + RESET
                )

            # Percentage
            percentage = f"{100 * frac:3.0f}%"

            # Timing
            rate = n / elapsed if elapsed > 0 else 0
            remaining = (total - n) / rate if rate > 0 else 0

            elapsed_str = self.format_interval(elapsed)
            remaining_str = self.format_interval(remaining) if rate > 0 else "??:??"

            return (
                f"{percentage}|{bar}| "
                f"[{n}/{total}] "
                f"[{elapsed_str}<{remaining_str}]"
            )
        else:
            return super().format_meter(n, total, elapsed, **kwargs)

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
