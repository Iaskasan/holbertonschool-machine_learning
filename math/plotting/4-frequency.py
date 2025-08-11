#!/usr/bin/env python3
"""Frequency task"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Generate a histogram with specific bins size"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 101, 10)
    y = np.arange(0, 31, 5)
    bin_width = np.arange(0, 101, 10)
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(x)
    plt.yticks(y)
    plt.hist(student_grades, edgecolor="black", bins=bin_width)
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()
