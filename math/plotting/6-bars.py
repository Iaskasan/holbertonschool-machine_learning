#!/usr/bin/env python3
"""Stacking Bars task"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Generate a stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    names = ["Farrah", "Fred", "Felicia"]
    x = np.arange(0, 3, 1)
    plt.figure(figsize=(6.4, 4.8))
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.yticks(np.arange(0, 81, 10))
    plt.xticks(x, names)
    plt.ylim(0, 80)
    plt.bar(x, apples, color="red", width=0.5, label="apples")
    plt.bar(x, bananas, bottom=apples, color="yellow",
            width=0.5, label="bananas")
    plt.bar(x, oranges, bottom=apples+bananas, color="#ff8000",
            width=0.5, label="oranges")
    plt.bar(x, peaches, bottom=apples+bananas+oranges, color="#ffe5b4",
            width=0.5, label="peaches")
    plt.legend()
    plt.show()

bars()