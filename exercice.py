#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import cmath
from scipy.integrate import quad
from matplotlib import pyplot
import math


# TODO: Définissez vos fonctions ici (il en manque quelques unes)


def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    # x = cartesian_coordinates[0]
    # y = cartesian_coordinates[1]
    # r = (x**2 + y**2)**1/2
    # angle = math.atan2(y, x)
    #
    # return np.array([r, angle])

    # ou encore

    # voir screenhot 9 nov
    # a = np.zeros([len(cartesian_coordinates), 2])
    #
    # for i in range(len(cartesian_coordinates)):
    #     a[i] = cmath.polar(cartesian_coordinates[i])
    #
    # return a

    # ou bien

    # results = []
    # for coords in cartesian_coordinates:
    #     results.append(cmath.polar(coords))
    #
    # return np.array(results)

    # et le best

    return np.array([cmath.polar(coord) for coord in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    # voir goated reponse screenshot du 9 novembre

    # ou sinon

    return np.abs(values - number).argmin()
    # argmin retourne lindex de la plus petite valeur


def graph():
    x = np.linspace(-1, 1, 250)
    y = x ** 2 * np.sin(1 / x ** 2) + x  # ici x est deja un array, et on cree un nouveau array pour y
    pyplot.plot(x, y)
    pyplot.show()
    # sinon voir screenshot 11nov pour avoir vrm des points et pas une ligne
    # sinon on a optimise le screenshot avec np.sin au lieu de math.sin, cest plus rapide et numpy est fait pour ca


def integrale_graph():
    f = lambda x: math.e ** (-x ** 2)

    x = np.linspace(-4, 4, 100)
    y = f(x)
    pyplot.plot(x, y, "r")
    pyplot.fill_between(x, y)
    pyplot.show()


def integrale():
    f = lambda x: np.e ** (-1 * x ** 2)
    return quad(f, -np.inf, np.inf)


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    integrale_graph()
    print(integrale())
