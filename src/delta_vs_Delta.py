# plots \delta = \sqrt{\Delta / 2}

import numpy as np, seaborn as sns, matplotlib.pyplot as plt, matplotlib._color_data as mcd


def plot():
    print("Plotting delta v/s Delta trend")
    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.markersize"] = 0.75
    plt.rcParams["text.usetex"] = True
    COLORS = [k for k in mcd.TABLEAU_COLORS.keys()]
    PATTERNS = ["d:", "x-", "*-", "o-", ".-", "d--", "x-.", ",--", "*:"]
    plt.figure(figsize=(6, 4))

    x = np.linspace(0, 0.75, num=1000)
    y = np.sqrt(x / 2)
    plt.ylim(-0.015, 1.015)
    plt.plot(
        x,
        y,
        # marker=PATTERNS[0][0],
        # linestyle=PATTERNS[0][1:],
        # alpha=0.7,
        color=COLORS[0],
        lw=1.0,
    )
    plt.xlabel(r"$\Delta$", fontsize=17)
    plt.ylabel(r"$\delta$", fontsize=17)
    plt.title(r"Plot of $\delta = \sqrt{\Delta / 2}$", fontsize=17)
    plt.tight_layout()
    plt.savefig(f"./figures/rq3/BLAU_sensitivity.png")
    plt.savefig(f"./figures/rq3/BLAU_sensitivity.pdf")
    plt.close()


if __name__ == "__main__":
    plot()
