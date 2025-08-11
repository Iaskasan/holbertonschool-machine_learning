#!/usr/bin/env python3
"""All in one task"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def all_in_one():
    """Merge all 5 previous graphs into one big figure"""

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

    fig = plt.figure(figsize=(6.4, 4.8))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    fig.suptitle("All in one")
    line = fig.add_subplot(gs[0, 0])
    scatter = fig.add_subplot(gs[0, 1])
    c14_decay = fig.add_subplot(gs[1, 0])
    two = fig.add_subplot(gs[1, 1])
    freq = fig.add_subplot(gs[2, :])

    line.plot(y0, color="red")
    line.set_xlim(0, 10)

    scatter.scatter(x1, y1, color="magenta")
    scatter.set_title("Men's Height vs Weight", fontsize="x-small")
    scatter.set_xlabel("Height (in)", fontsize="x-small")
    scatter.set_ylabel("Weight (lbs)", fontsize="x-small")

    c14_decay.set_title("Exponential Decay of C-14", fontsize="x-small")
    c14_decay.set_xlabel("Time (years)", fontsize="x-small")
    c14_decay.set_ylabel("Fraction Remaining", fontsize="x-small")
    c14_decay.plot(x2, y2)
    c14_decay.set_yscale("log")
    c14_decay.set_xlim(0, 28650)

    two.set_title("Exponential Decay of Radioactive Elements",
                  fontsize="x-small")
    two.set_xlabel("Time (years)", fontsize="x-small")
    two.set_ylabel("Fraction Remaining", fontsize="x-small")
    two.plot(x3, y31, label="C-14", linestyle="--", color="red")
    two.plot(x3, y32, label="Ra-226", color="green")
    two.set_xlim(0, 20000)
    two.set_ylim(0, 1)
    two.legend()

    freq.set_title("Project A", fontsize="x-small")
    freq.set_xlabel("Grades", fontsize="x-small")
    freq.set_ylabel("Number of Students", fontsize="x-small")
    freq.set_xticks(np.arange(0, 101, 10))
    freq.set_yticks(np.arange(0, 31, 10))
    freq.set_xlim(0, 100)
    freq.set_ylim(0, 30)
    freq.hist(student_grades, edgecolor="black", bins=np.arange(0, 101, 10))

    plt.tight_layout()
    plt.show()
