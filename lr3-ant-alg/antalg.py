#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python 3.12.6

import os
import random
import math

# Constants
MAX_CITIES = 20
MAX_ANTS = 10
MAX_DISTANCE = 100
INIT_PHEROMONE = 1.0
ALPHA = 1.0
BETA = 2.0
RHO = 0.5
QVAL = 100
MAX_TOUR = float('inf')
MAX_TIME = 1000

# Data structures


class City:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Ant:
    def __init__(self):
        self.curCity = 0
        self.nextCity = -1
        self.tourLength = 0.0
        self.path = [-1] * MAX_CITIES
        self.pathIndex = 0
        self.tabu = [0] * MAX_CITIES


cities = [City() for _ in range(MAX_CITIES)]
ants = [Ant() for _ in range(MAX_ANTS)]
distance = [[0.0] * MAX_CITIES for _ in range(MAX_CITIES)]
pheromone = [[INIT_PHEROMONE] * MAX_CITIES for _ in range(MAX_CITIES)]
best = MAX_TOUR
bestIndex = 0


def getRand(max_value):
    return random.randint(0, max_value)


def getSRand():
    return random.random()


def init():
    global best, bestIndex
    best = MAX_TOUR
    bestIndex = 0

    for from_city in range(MAX_CITIES):
        cities[from_city].x = getRand(MAX_DISTANCE)
        cities[from_city].y = getRand(MAX_DISTANCE)
        for to_city in range(MAX_CITIES):
            distance[from_city][to_city] = 0.0
            pheromone[from_city][to_city] = INIT_PHEROMONE

    for from_city in range(MAX_CITIES):
        for to_city in range(MAX_CITIES):
            if from_city != to_city and distance[from_city][to_city] == 0.0:
                xd = abs(cities[from_city].x - cities[to_city].x)
                yd = abs(cities[from_city].y - cities[to_city].y)
                distance[from_city][to_city] = math.sqrt(xd * xd + yd * yd)
                distance[to_city][from_city] = distance[from_city][to_city]

    to_city = 0
    for ant in ants:
        if to_city == MAX_CITIES:
            to_city = 0
        ant.curCity = to_city
        to_city += 1
        ant.path = [-1] * MAX_CITIES
        ant.pathIndex = 1
        ant.path[0] = ant.curCity
        ant.nextCity = -1
        ant.tourLength = 0.0
        ant.tabu = [0] * MAX_CITIES
        ant.tabu[ant.curCity] = 1


def restartAnts():
    global best, bestIndex
    to_city = 0
    for ant in ants:
        if ant.tourLength < best:
            best = ant.tourLength
            bestIndex = ants.index(ant)
        ant.nextCity = -1
        ant.tourLength = 0.0
        ant.path = [-1] * MAX_CITIES
        ant.pathIndex = 1
        if to_city == MAX_CITIES:
            to_city = 0
        ant.curCity = to_city
        to_city += 1
        ant.path[0] = ant.curCity
        ant.tabu = [0] * MAX_CITIES
        ant.tabu[ant.curCity] = 1


def antProduct(from_city, to_city):
    return (pheromone[from_city][to_city] ** ALPHA) * ((1.0 / distance[from_city][to_city]) ** BETA)


def selectNextCity(ant_index):
    from_city = ants[ant_index].curCity
    denom = sum(antProduct(from_city, to_city) for to_city in range(
        MAX_CITIES) if ants[ant_index].tabu[to_city] == 0)
    assert denom != 0.0

    while True:
        to_city = random.randint(0, MAX_CITIES - 1)
        if ants[ant_index].tabu[to_city] == 0:
            p = antProduct(from_city, to_city) / denom
            if getSRand() < p:
                break
    return to_city


def simulateAnts():
    moving = 0
    for k in range(MAX_ANTS):
        if ants[k].pathIndex < MAX_CITIES:
            ants[k].nextCity = selectNextCity(k)
            ants[k].tabu[ants[k].nextCity] = 1
            ants[k].path[ants[k].pathIndex] = ants[k].nextCity
            ants[k].pathIndex += 1
            ants[k].tourLength += distance[ants[k].curCity][ants[k].nextCity]
            if ants[k].pathIndex == MAX_CITIES:
                ants[k].tourLength += distance[ants[k].path[MAX_CITIES - 1]
                                               ][ants[k].path[0]]
            ants[k].curCity = ants[k].nextCity
            moving += 1
    return moving


def updateTrails():
    for from_city in range(MAX_CITIES):
        for to_city in range(MAX_CITIES):
            if from_city != to_city:
                pheromone[from_city][to_city] *= (1.0 - RHO)
                if pheromone[from_city][to_city] < 0.0:
                    pheromone[from_city][to_city] = INIT_PHEROMONE

    for ant in ants:
        for i in range(MAX_CITIES):
            if i < MAX_CITIES - 1:
                from_city = ant.path[i]
                to_city = ant.path[i + 1]
            else:
                from_city = ant.path[i]
                to_city = ant.path[0]
            pheromone[from_city][to_city] += (QVAL / ant.tourLength)
            pheromone[to_city][from_city] = pheromone[from_city][to_city]

    for from_city in range(MAX_CITIES):
        for to_city in range(MAX_CITIES):
            pheromone[from_city][to_city] *= RHO


def emitDataFile(ant_index):
    with open("out/cities.dat", "w") as fp:
        for city in cities:
            fp.write(f"{city.x} {city.y}\n")

    with open("out/solution.dat", "w") as fp:
        for city_index in ants[ant_index].path:
            fp.write(f"{cities[city_index].x} {cities[city_index].y}\n")
        fp.write(f"{cities[ants[ant_index].path[0]].x} {
                 cities[ants[ant_index].path[0]].y}\n")


def emitTable():
    for from_city in range(MAX_CITIES):
        for to_city in range(MAX_CITIES):
            print(f"{pheromone[from_city][to_city]:5.2g} ", end="")
        print()
    print()


def main():
    curTime = 0
    random.seed()

    init()

    # if out directory does not exist, create it
    if not os.path.exists("out"):
        os.makedirs("out")

    while curTime < MAX_TIME:
        curTime += 1
        if simulateAnts() == 0:
            updateTrails()
            if curTime != MAX_TIME:
                restartAnts()
            print(f"Time is {curTime} ({best})")

    print(f"best tour {best}\n\n")
    emitDataFile(bestIndex)


if __name__ == "__main__":
    main()
