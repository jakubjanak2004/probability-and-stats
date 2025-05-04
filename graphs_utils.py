import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def generate_graphs_for_data(data, hist_bins=20):
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    # 1. Histogram
    axs[0].hist(data, bins=hist_bins, edgecolor='black', alpha=0.7)
    axs[0].set_title("Histogram of Data")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    axs[0].grid(axis='y', linestyle='--', alpha=0.6)

    # 2. Boxplot
    axs[1].boxplot(data, vert=True)
    axs[1].set_title("Boxplot of Data")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    # 3. Empirical Distribution Function (ECDF)
    ecdf = ECDF(data)
    axs[2].step(ecdf.x, ecdf.y, where='post', label='ECDF')
    axs[2].set_title("Empirical Distribution Function (EDF)")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("ECDF")
    axs[2].grid(True)
    axs[2].legend()
    plt.show()


def plot_pdf_cdf(f_x, F_x, args_for_f=(), args_for_F=(), values_for_f=np.array([]), values_for_F=np.array([]),
                 title="Distribution"):
    x = values_for_f

    # Evaluate with optional parameters
    y = f_x(values_for_f, *args_for_f)
    Fy = F_x(values_for_F, *args_for_F)

    # Split for clean PDF rendering if needed
    x_neg = x[x < 0]
    y_neg = y[x < 0]
    x_pos = x[x >= 0]
    y_pos = y[x >= 0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 9))

    # PDF
    axs[0].plot(x_neg, y_neg, color='blue')
    axs[0].plot(x_pos, y_pos, color='blue')
    axs[0].set_ylim(bottom=-0.1)
    axs[0].set_title(f"{title} PDF")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("f(x)")
    axs[0].grid(True)

    # CDF
    axs[1].plot(x, Fy, color='green')
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_title(f"{title} CDF")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("F(x)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_prob_cdf(f_x, F_x, args_for_f=(), args_for_F=(), values_for_f=np.array([]), values_for_F=np.array([]),
                  title="Distribution"):
    x = values_for_f

    # Evaluate with optional parameters
    y = f_x(values_for_f, *args_for_f)
    Fy = F_x(values_for_F, *args_for_F)

    fig, axs = plt.subplots(2, 1, figsize=(8, 9))

    # Probabilities
    axs[0].bar(x, y, color='blue')
    axs[0].set_title(f"{title} Probabilities")
    axs[0].set_xlabel("k")
    axs[0].set_ylabel("P(X = k)")
    axs[0].grid(True)

    # CDF
    axs[1].step(x, Fy, where='post', color='green')
    axs[1].set_title(f"{title} CDF")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("F(x)")
    axs[1].grid(True)

    # plt.tight_layout()
    plt.show()
