#!/usr/bin/env python3
"""plot of a histogram of student scores for a project"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """histogram of students and grades"""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bin = np.arange(0, 101, 10)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=bin, edgecolor="black")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(bin)
    plt.title("Project A")
    plt.show()
