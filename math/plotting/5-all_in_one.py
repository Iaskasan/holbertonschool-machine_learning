#!/usr/bin/env python3
"""Plot of all 5 previous plots all in one"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """Gathering of all previous plots"""

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bin = np.arange(0, 101, 10)

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("All in One")

    line = fig.add_subplot(gs[0, 0])
    scatter = fig.add_subplot(gs[0, 1])
    scale = fig.add_subplot(gs[1, 0])
    two = fig.add_subplot(gs[1, 1])
    freq = fig.add_subplot(gs[2, :])

    line.plot(y0, "r")
    line.set_xlim(0, 10)

    scatter.scatter(x1, y1, color="magenta")
    scatter.set_xlabel("Height (in)", size="x-small")
    scatter.set_ylabel("Weight (lbs)", size="x-small")
    scatter.set_title("Men's Height vs Weight", size="x-small")

    scale.plot(x2, y2)
    scale.set_yscale("log")
    scale.set_xlim(0, 28650)
    scale.set_xlabel("Time (years)", size="x-small")
    scale.set_ylabel("Fraction Remaining", size="x-small")
    scale.set_title("Exponential Decay of C-14", size="x-small")

    two.plot(x3, y31, "r--", label="C-14")
    two.plot(x3, y32, "g", label="Ra-226")
    two.set_xlabel("Time (years)", size="x-small")
    two.set_ylabel("Fraction Remaining", size="x-small")
    two.set_title("Exponential Decay of Radioactive Elements", size="x-small")
    two.legend()
    two.set_xlim(0, 20000)
    two.set_ylim(0, 1)

    freq.hist(student_grades, bins=bin, edgecolor="black")
    freq.set_xlabel("Grades", size="x-small")
    freq.set_ylabel("Number of Students", size="x-small")
    freq.set_xlim(0, 100)
    freq.set_ylim(0, 30)
    freq.set_xticks(bin)
    freq.set_title("Project A", size="x-small")

    plt.tight_layout()
    plt.show()
