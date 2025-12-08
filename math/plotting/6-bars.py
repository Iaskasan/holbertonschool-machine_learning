#!/usr/bin/env python3
"""Stacked bars method"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Stacked bars plot that represents
    number of fruits per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    print(fruit)
    plt.figure(figsize=(6.4, 4.8))
    names = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    fruits = {
        "apples": fruit[0],
        "bananas": fruit[1],
        "oranges": fruit[2],
        "peaches": fruit[3]
    }
    bottom = np.zeros(3)
    for (fruittype, values), colors in zip((fruits.items()), colors):
        plt.bar(names, values, label=fruittype,
                bottom=bottom, color=colors, width=0.5)
        bottom += values
    plt.ylim(0, 80)
    plt.legend()
    plt.ylabel("Quandity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.yticks(np.arange(0, 81, 10))
    plt.show()
