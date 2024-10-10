#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python 3.12.6

import os
import random
import math
from pprint import pprint
from tabulate import tabulate
from matplotlib import pyplot as plt
import networkx as nx

# Constants
MAX_CITIES = 20
MAX_DISTANCE = 100
INIT_PHEROMONE = 0

# Data structures


class City:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Ant:
    def __init__(self, max_cities):
        self.curCity = 0
        self.nextCity = -1
        self.tourLength = 0.0
        self.path = [-1] * max_cities
        self.pathIndex = 0
        self.tabu = [0] * max_cities


def getRand(max_value):
    return random.randint(0, max_value)


def getSRand():
    return random.random()


def init(cities, ants, distance, pheromone, max_cities, max_distance, init_pheromone):
    for from_city in range(max_cities):
        cities[from_city].x = getRand(max_distance)
        cities[from_city].y = getRand(max_distance)
        for to_city in range(max_cities):
            distance[from_city][to_city] = 0.0
            pheromone[from_city][to_city] = init_pheromone

    for from_city in range(max_cities):
        for to_city in range(max_cities):
            if from_city != to_city and distance[from_city][to_city] == 0.0:
                xd = abs(cities[from_city].x - cities[to_city].x)
                yd = abs(cities[from_city].y - cities[to_city].y)
                distance[from_city][to_city] = math.sqrt(xd * xd + yd * yd)
                distance[to_city][from_city] = distance[from_city][to_city]

    to_city = 0
    for ant in ants:
        if to_city == max_cities:
            to_city = 0
        ant.curCity = to_city
        to_city += 1
        ant.path = [-1] * max_cities
        ant.pathIndex = 1
        ant.path[0] = ant.curCity
        ant.nextCity = -1
        ant.tourLength = 0.0
        ant.tabu = [0] * max_cities
        ant.tabu[ant.curCity] = 1


def restartAnts(ants, best, bestIndex, max_cities):
    to_city = 0
    for ant in ants:
        if ant.tourLength < best:
            best = ant.tourLength
            bestIndex = ants.index(ant)
        ant.nextCity = -1
        ant.tourLength = 0.0
        ant.path = [-1] * max_cities
        ant.pathIndex = 1
        if to_city == max_cities:
            to_city = 0
        ant.curCity = to_city
        to_city += 1
        ant.path[0] = ant.curCity
        ant.tabu = [0] * max_cities
        ant.tabu[ant.curCity] = 1
    return best, bestIndex


def antProduct(from_city, to_city, pheromone, distance, alpha, beta):
    return (pheromone[from_city][to_city] ** alpha) * ((1.0 / distance[from_city][to_city]) ** beta)


def selectNextCity(ant_index, ants, pheromone, distance, alpha, beta, max_cities):
    from_city = ants[ant_index].curCity
    denom = sum(antProduct(from_city, to_city, pheromone, distance, alpha, beta)
                for to_city in range(max_cities) if ants[ant_index].tabu[to_city] == 0)
    assert denom != 0.0

    while True:
        to_city = random.randint(0, max_cities - 1)
        if ants[ant_index].tabu[to_city] == 0:
            p = antProduct(from_city, to_city, pheromone,
                           distance, alpha, beta) / denom
            if getSRand() < p:
                break
    return to_city


def simulateAnts(ants, pheromone, distance, alpha, beta, max_cities):
    moving = 0
    for k in range(len(ants)):
        if ants[k].pathIndex < max_cities:
            ants[k].nextCity = selectNextCity(
                k, ants, pheromone, distance, alpha, beta, max_cities)
            ants[k].tabu[ants[k].nextCity] = 1
            ants[k].path[ants[k].pathIndex] = ants[k].nextCity
            ants[k].pathIndex += 1
            ants[k].tourLength += distance[ants[k].curCity][ants[k].nextCity]
            if ants[k].pathIndex == max_cities:
                ants[k].tourLength += distance[ants[k].path[max_cities - 1]
                                               ][ants[k].path[0]]
            ants[k].curCity = ants[k].nextCity
            moving += 1
    return moving


def updateTrails(ants, pheromone, distance, rho, qval, max_cities):
    global INIT_PHEROMONE
    for from_city in range(max_cities):
        for to_city in range(max_cities):
            if from_city != to_city:
                pheromone[from_city][to_city] *= (1.0 - rho)
                if pheromone[from_city][to_city] < 0.0:
                    pheromone[from_city][to_city] = INIT_PHEROMONE

    for ant in ants:
        for i in range(max_cities):
            if i < max_cities - 1:
                from_city = ant.path[i]
                to_city = ant.path[i + 1]
            else:
                from_city = ant.path[i]
                to_city = ant.path[0]
            pheromone[from_city][to_city] += (qval / ant.tourLength)
            pheromone[to_city][from_city] = pheromone[from_city][to_city]

    for from_city in range(max_cities):
        for to_city in range(max_cities):
            pheromone[from_city][to_city] *= rho


def emitDataFile(cities, ants, ant_index):
    with open("out/cities.dat", "w") as fp:
        for city in cities:
            fp.write(f"{city.x} {city.y}\n")

    with open("out/solution.dat", "w") as fp:
        for city_index in ants[ant_index].path:
            fp.write(f"{cities[city_index].x} {cities[city_index].y}\n")
        fp.write(f"{cities[ants[ant_index].path[0]].x} {
                 cities[ants[ant_index].path[0]].y}\n")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

def main():
    # max_cities = int(input("Enter the number of cities: "))
    # max_ants = int(input("Enter the number of ants: "))
    # max_distance = int(input("Enter the maximum distance: "))
    # init_pheromone = float(input("Enter the initial pheromone level: "))
    # alpha = float(input("Enter the alpha value: "))
    # beta = float(input("Enter the beta value: "))
    # rho = float(input("Enter the rho value: "))
    # qval = float(input("Enter the Q value: "))
    # max_time = int(input("Enter the maximum time: "))

    max_cities = 20
    max_ants = 10
    max_distance = 100
    init_pheromone = 1.0
    alpha = 1.0
    beta = 5.0
    rho = 0.5
    qval = 100
    max_time = 100

    cities = [City() for _ in range(max_cities)]
    ants = [Ant(max_cities) for _ in range(max_ants)]
    distance = [[0.0] * max_cities for _ in range(max_cities)]
    pheromone = [[init_pheromone] * max_cities for _ in range(max_cities)]
    best = float('inf')
    bestIndex = 0
    global INIT_PHEROMONE
    INIT_PHEROMONE = init_pheromone

    random.seed()
    init(cities, ants, distance, pheromone,
         max_cities, max_distance, init_pheromone)

    if not os.path.exists("out"):
        os.makedirs("out")

    curTime = 0
    while curTime < max_time:
        curTime += 1
        if simulateAnts(ants, pheromone, distance, alpha, beta, max_cities) == 0:
            updateTrails(ants, pheromone, distance, rho, qval, max_cities)
            if curTime != max_time:
                best, bestIndex = restartAnts(
                    ants, best, bestIndex, max_cities)
            if curTime % 10 == 0:
                # plot the pheromone matrix as networkx graph
                G = nx.random_geometric_graph(max_cities, radius=0.4, seed=69)
                pos = nx.get_node_attributes(G, "pos")
                # resize plot
                fig = plt.figure()
                for i in range(max_cities):
                    G.add_node(i)
                for i in range(max_cities):
                    for j in range(i + 1, max_cities):
                        G.add_edge(i, j, weight=pheromone[i][j])

                # Extract weights
                weights = [G[u][v]['weight'] for u, v in G.edges()]


                # Draw the graph with weights as edge widths
                nx.draw(G, pos, with_labels=True, width=weights)

                # pixels = plt.gcf().canvas.get_renderer().buffer_rgba()
                # save pixels
                # save_bmp(pixels, 800, 800, "out/antalg.bmp")
                canvas = FigureCanvasAgg(fig)
                canvas.draw()

                width, height = fig.get_size_inches() * fig.get_dpi()
                pixel_data = canvas.buffer_rgba() # r, g, b, a

                # Save the image to a file
                image = Image.frombytes("RGBA", (int(width), int(height)), pixel_data)
                image.save("out/antalg.png")

            print(f"Time is {curTime} ({best})")

    print(f"Best tour length: {best}")
    emitDataFile(cities, ants, bestIndex)


if __name__ == "__main__":
    main()
