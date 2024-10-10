#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python 3.12.6

import dearpygui.dearpygui as dpg
import networkx as nx
from matplotlib import pyplot as plt
from tabulate import tabulate
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import os
import random
import math
import threading
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import use as use_backend
use_backend('Agg')

# Global variables
process = None
is_running = False
init_pheromone = 0

data_blob = {
    'max_cities': None,
    'max_ants': None,
    'max_distance': None,
    'init_pheromone': None,
    'alpha': None,
    'beta': None,
    'rho': None,
    'qval': None,
    'max_time': None
}

best_length_y = []
best_length_x = []

start_time = 0

# Constants
TEXTURE_WIDTH = 500
TEXTURE_HEIGHT = 500


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
    try:
        return (pheromone[from_city][to_city] ** alpha) * ((1.0 / distance[from_city][to_city]) ** beta)
    except ZeroDivisionError:
        return 0.0


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
    global init_pheromone
    for from_city in range(max_cities):
        for to_city in range(max_cities):
            if from_city != to_city:
                pheromone[from_city][to_city] *= (1.0 - rho)
                if pheromone[from_city][to_city] < 0.0:
                    pheromone[from_city][to_city] = init_pheromone

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
    global init_pheromone, TEXTURE_WIDTH, TEXTURE_HEIGHT, is_running, data_blob
    # max_cities = 20
    # max_ants = 10
    # max_distance = 100
    # init_pheromone = 1.0
    # alpha = 1.0
    # beta = 5.0
    # rho = 0.5
    # qval = 100
    # max_time = 100
    print(data_blob)

    cities = [City() for _ in range(data_blob['max_cities'])]
    ants = [Ant(data_blob['max_cities']) for _ in range(data_blob['max_ants'])]
    distance = [[0.0] * data_blob['max_cities']
                for _ in range(data_blob['max_cities'])]
    pheromone = [[init_pheromone] * data_blob['max_cities']
                 for _ in range(data_blob['max_cities'])]
    best = float('inf')
    bestIndex = 0
    init_pheromone = data_blob['init_pheromone']

    random.seed()
    init(cities, ants, distance, pheromone,
         data_blob['max_cities'], data_blob['max_distance'], init_pheromone)

    if not os.path.exists("out"):
        os.makedirs("out")

    curTime = 0
    while curTime < data_blob['max_time']:
        if not is_running:
            break
        curTime += 1
        if simulateAnts(ants, pheromone, distance, data_blob['alpha'], data_blob['beta'], data_blob['max_cities']) == 0:
            updateTrails(ants, pheromone, distance,
                         data_blob['rho'], data_blob['qval'], data_blob['max_cities'])
            if curTime != data_blob['max_time']:
                best, bestIndex = restartAnts(
                    ants, best, bestIndex, data_blob['max_cities'])
            if curTime % 1 == 0:
                cmap = LinearSegmentedColormap.from_list(
                    'custom_cmap', ['#FFA500', '#FF0000', '#00FF00'])
                # plot the pheromone matrix as networkx graph
                G = nx.random_geometric_graph(
                    data_blob['max_cities'], radius=0.4, seed=data_blob['networkx_seed'])
                pos = nx.get_node_attributes(G, "pos")
                fig = plt.figure()
                # resize plot in pixels
                fig.set_size_inches(
                    TEXTURE_WIDTH / fig.get_dpi(), TEXTURE_WIDTH / fig.get_dpi())
                for i in range(data_blob['max_cities']):
                    G.add_node(i)
                for i in range(data_blob['max_cities']):
                    for j in range(i + 1, data_blob['max_cities']):
                        G.add_edge(i, j, weight=pheromone[i][j])

                # Extract weights
                weights = [G[u][v]['weight'] for u, v in G.edges()]

                # Normalize weights to range [0, 1]
                norm_weights = np.array(weights) / max(weights)

                # Map normalized weights to colors
                edge_colors = [cmap(w) for w in norm_weights]

                # Draw the graph with weights as edge widths
                nx.draw(G, pos, with_labels=True, width=weights *
                        20, edge_color=edge_colors)

                # pixels = plt.gcf().canvas.get_renderer().buffer_rgba()
                # save pixels
                # save_bmp(pixels, 800, 800, "out/antalg.bmp")
                canvas = FigureCanvasAgg(fig)
                canvas.draw()

                width, height = fig.get_size_inches() * fig.get_dpi()
                pixel_data = canvas.buffer_rgba()  # r, g, b, a

                # Save the image to a file
                # image = Image.frombytes(
                #     "RGBA", (int(width), int(height)), pixel_data)
                # image.save("out/antalg.png")

                # normalize pixel data to 0-1
                pixel_data = [x / 255 for x in bytearray(pixel_data)]

                dpg.set_value("texture_tag", pixel_data)
                plt.close(fig)

            print(f"Time is {curTime} ({best})")

    print(f"Best tour length: {best}")

    if -1 in ants[bestIndex].path:
        # if -1 in solution, print error
        print("Solution was not found")
    else:
        # print best solution as 12 > 23 > 234 > ...
        print("Best solution:")
        for city_index in ants[bestIndex].path:
            print(f"{city_index} > ", end="")
        print(f"{ants[bestIndex].path[0]}")
        emitDataFile(cities, ants, bestIndex)
    stop()


def update_layout():
    port_width = dpg.get_viewport_client_width()
    port_height = dpg.get_viewport_client_height()

    inputs_height = 26

    image_size = min(port_width, port_height - inputs_height * 10)

    # global texture_width, texture_height
    # texture_width = image_size / 4
    # texture_height = image_size / 4

    plots_width = port_width - image_size - 20
    plots_count = 1
    plots_height = image_size / plots_count - plots_count + 1  # / 3.031

    inputs_width = min(max(port_width * 0.15, 100), 500)

    outputs_width = min(max(port_width * 0.15, 100), 500)

    if dpg.does_item_exist("image"):
        dpg.configure_item("image", width=image_size, height=image_size)

    if dpg.does_item_exist("num_cities"):
        dpg.configure_item("num_cities", width=inputs_width)
    if dpg.does_item_exist("num_ants"):
        dpg.configure_item("num_ants", width=inputs_width)
    if dpg.does_item_exist("max_distance"):
        dpg.configure_item("max_distance", width=inputs_width)
    if dpg.does_item_exist("init_pheromone"):
        dpg.configure_item("init_pheromone", width=inputs_width)
    if dpg.does_item_exist("alpha"):
        dpg.configure_item("alpha", width=inputs_width)
    if dpg.does_item_exist("beta"):
        dpg.configure_item("beta", width=inputs_width)
    if dpg.does_item_exist("rho"):
        dpg.configure_item("rho", width=inputs_width)
    if dpg.does_item_exist("qval"):
        dpg.configure_item("qval", width=inputs_width)
    if dpg.does_item_exist("max_time"):
        dpg.configure_item("max_time", width=inputs_width)


    if dpg.does_item_exist("temp_text"):
        dpg.configure_item("temp_text", width=outputs_width)
    if dpg.does_item_exist("energy_text"):
        dpg.configure_item("energy_text", width=outputs_width)
    if dpg.does_item_exist("bads_text"):
        dpg.configure_item("bads_text", width=outputs_width)
    if dpg.does_item_exist("time_text"):
        dpg.configure_item("time_text", width=outputs_width)
    if dpg.does_item_exist("networkx_seed"):
        dpg.configure_item("networkx_seed", width=outputs_width)

    if dpg.does_item_exist("plot_best_length"):
        dpg.configure_item("plot_best_length",
                           width=plots_width, height=plots_height)
    # if dpg.does_item_exist("plot_energy"):
    #     dpg.configure_item("plot_energy", width=plots_width,
    #                        height=plots_height)
    # if dpg.does_item_exist("plot_bads"):
    #     dpg.configure_item("plot_bads", width=plots_width, height=plots_height)


def read_data_blob_from_ui():
    global data_blob
    data_blob['max_cities'] = dpg.get_value("num_cities")
    data_blob['max_ants'] = dpg.get_value("num_ants")
    data_blob['max_distance'] = dpg.get_value("max_distance")
    data_blob['init_pheromone'] = dpg.get_value("init_pheromone")
    data_blob['alpha'] = dpg.get_value("alpha")
    data_blob['beta'] = dpg.get_value("beta")
    data_blob['rho'] = dpg.get_value("rho")
    data_blob['qval'] = dpg.get_value("qval")
    data_blob['max_time'] = dpg.get_value("max_time")
    data_blob['networkx_seed'] = dpg.get_value("networkx_seed")


def run():
    global is_running
    global process
    stop()
    if process is None:
        is_running = True
        read_data_blob_from_ui()
        process = threading.Thread(target=main)
        process.start()


def stop():
    global is_running
    global process
    is_running = False
    if process is not None:
        try:
            process.join()
        except:
            pass
    process = None


def gen_texture_empty(width, height, scale=1):
    texture = []
    for y in range(height):
        for x in range(width):
            if ((x // scale) + (y // scale)) % 2 == 0:
                texture.extend([0.5, 0.0, 0.5, 1.0])  # Purple
            else:
                texture.extend([0.0, 0.0, 0.0, 1.0])  # Black
    return texture


def gen_texture_solid(width: int, height: int, color: list) -> list:
    """Generate a solid color texture.

    Args:
        width (int): Width of the texture.
        height (int): Height of the texture.
        color (list): List of RGBA values. 
        For example, [1.0, 0.0, 0.0, 1.0] is red.

    Returns:
        list: List of RGBA values.
    """
    texture = []
    for y in range(height):
        for x in range(width):
            texture.extend(color)
    return texture


def app():
    global TEXTURE_WIDTH, TEXTURE_HEIGHT, data_blob
    dpg.create_context()
    dpg.create_viewport(title='Ant algorithm', width=800, height=600)
    dpg.set_viewport_resize_callback(update_layout)
    # print(dpg.get_viewport_client_width())
    # print(dpg.get_viewport_client_height())

    # width, height, channels, data = dpg.load_image("solution.png")
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width=TEXTURE_WIDTH, height=TEXTURE_HEIGHT, default_value=gen_texture_solid(TEXTURE_WIDTH, TEXTURE_HEIGHT, [1, 1, 1, 1]), tag="texture_tag")

    with dpg.window(label="Example Window", tag="fullscreen"):

        with dpg.theme(tag="plot_theme_temp"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (255, 99, 99),
                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None,
                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_MarkerSize, 0,
                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3.0,
                    category=dpg.mvThemeCat_Plots)
        # with dpg.theme(tag="plot_theme_energy"):
        #     with dpg.theme_component(dpg.mvLineSeries):
        #         dpg.add_theme_color(
        #             dpg.mvPlotCol_Line, (22, 90, 255),
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None,
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_MarkerSize, 0,
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_LineWeight, 3.0,
        #             category=dpg.mvThemeCat_Plots)
        # with dpg.theme(tag="plot_theme_bads"):
        #     with dpg.theme_component(dpg.mvLineSeries):
        #         dpg.add_theme_color(
        #             dpg.mvPlotCol_Line, (123, 123, 123),
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None,
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_MarkerSize, 0,
        #             category=dpg.mvThemeCat_Plots)
        #         dpg.add_theme_style(
        #             dpg.mvPlotStyleVar_LineWeight, 3.0,
        #             category=dpg.mvThemeCat_Plots)

        def on_value_change_num_cities(sender, app_data, user_data):
            stop()
            user_data['num_cities'] = int(app_data)

        def on_value_change_num_ants(sender, app_data, user_data):
            stop()
            user_data['num_ants'] = int(app_data)

        def on_value_change_max_distance(sender, app_data, user_data):
            stop()
            user_data['max_distance'] = int(app_data)

        def on_value_change_init_pheromone(sender, app_data, user_data):
            stop()
            user_data['init_pheromone'] = float(app_data)

        def on_value_change_alpha(sender, app_data, user_data):
            stop()
            user_data['alpha'] = float(app_data)

        def on_value_change_beta(sender, app_data, user_data):
            stop()
            user_data['beta'] = float(app_data)

        def on_value_change_rho(sender, app_data, user_data):
            stop()
            user_data['rho'] = float(app_data)

        def on_value_change_qval(sender, app_data, user_data):
            stop()
            user_data['qval'] = float(app_data)

        def on_value_change_max_time(sender, app_data, user_data):
            stop()
            user_data['max_time'] = int(app_data)

        def on_value_change_networkx_seed(sender, app_data, user_data):
            user_data['networkx_seed'] = int(app_data)

        # def on_value_change_alpha(sender, app_data, user_data):
        #     stop()
        #     user_data['alpha'] = float(app_data)

        # def on_value_change_steps(sender, app_data, user_data):
        #     stop()
        #     user_data['steps_per_change'] = int(app_data)

        # def on_value_change_num_queens(sender, app_data, user_data):
        #     stop()
        #     user_data['num_queens'] = int(app_data)
        #     # emit_solution_image(None)
        #     width, height, channels, data = dpg.load_image("solution.png")
        #     dpg.set_value("texture_tag", data)

        # Максимальная температура
        with dpg.group(horizontal=True):
            with dpg.group():

                # Max city count
                dpg.add_input_int(label="Number of cities",
                                  default_value=20,
                                  width=200,
                                  user_data=data_blob,
                                  callback=on_value_change_num_cities,
                                  min_value=1, max_value=999,
                                  tag="num_cities")
                # Max ant count
                dpg.add_input_int(label="Number of ants",
                                  default_value=10,
                                  width=200,
                                  user_data=data_blob,
                                  callback=on_value_change_num_ants,
                                  min_value=1, max_value=999,
                                  tag="num_ants")
                # Max distance
                dpg.add_input_int(label="Maximum distance",
                                  default_value=100,
                                  width=200,
                                  user_data=data_blob,
                                  callback=on_value_change_max_distance,
                                  min_value=1,
                                  tag="max_distance")
                # Initial pheromone
                dpg.add_input_float(label="Initial pheromone level",
                                    default_value=1.0,
                                    width=200,
                                    user_data=data_blob,
                                    callback=on_value_change_init_pheromone,
                                    min_value=0.0,
                                    step=0.01,
                                    tag="init_pheromone")
                # Alpha
                dpg.add_input_float(label="Alpha",
                                    default_value=1.0,
                                    width=200,
                                    user_data=data_blob,
                                    callback=on_value_change_alpha,
                                    min_value=0.0,
                                    step=0.01,
                                    tag="alpha")
                # Beta
                dpg.add_input_float(label="Beta",
                                    default_value=5.0,
                                    width=200,
                                    user_data=data_blob,
                                    callback=on_value_change_beta,
                                    min_value=0.0,
                                    step=0.01,
                                    tag="beta")
                # Rho
                dpg.add_input_float(label="Rho",
                                    default_value=0.5,
                                    width=200,
                                    user_data=data_blob,
                                    callback=on_value_change_rho,
                                    min_value=0.0,
                                    step=0.01,
                                    tag="rho")
                # Q value
                dpg.add_input_float(label="Q value",
                                    default_value=100,
                                    width=200,
                                    user_data=data_blob,
                                    callback=on_value_change_qval,
                                    min_value=0.0,
                                    step=0.01,
                                    tag="qval")
                # Max time
                dpg.add_input_int(label="Maximum time",
                                  default_value=100,
                                  width=200,
                                  user_data=data_blob,
                                  callback=on_value_change_max_time,
                                  min_value=1,
                                  tag="max_time")

            with dpg.group():
                # text output
                dpg.add_input_text(default_value="...",
                                   readonly=True,
                                   tag="temp_text",
                                   width=50)
                dpg.add_input_text(default_value="...",
                                   readonly=True,
                                   tag="energy_text",
                                   width=50)
                dpg.add_input_text(default_value="...",
                                   readonly=True,
                                   tag="bads_text",
                                   width=50)
                dpg.add_input_text(default_value="...",
                                   readonly=True,
                                   tag="time_text",
                                   width=50)
                dpg.add_input_int(label="Image seed",
                                  default_value=100,
                                  width=50,
                                  user_data=data_blob,
                                  callback=on_value_change_networkx_seed,
                                  min_value=1,
                                  tag="networkx_seed")

            with dpg.group():
                dpg.add_text("Temperature")
                dpg.add_text("Best Energy")
                dpg.add_text("Bad Solutions")
                dpg.add_text("Time")

        with dpg.group(horizontal=True):
            dpg.add_button(label="run", callback=run)
            dpg.add_button(label="stop", callback=stop)

        # global texture_width, texture_height
        text_width = dpg.get_viewport_client_width() * 5 / 10
        text_height = dpg.get_viewport_client_width() * 5 / 10
        with dpg.group(horizontal=True):
            dpg.add_image("texture_tag",
                          width=text_width,
                          height=text_height, tag="image")

            # emit_solution_image(None)
            # width, height, channels, data = dpg.load_image("solution.png")
            # dpg.set_value("texture_tag", data)

            plots_width = text_width / 2
            plots_height = text_height / 3.031  # 4.548
            with dpg.group():
                with dpg.plot(label="Best Length",
                              width=plots_width,
                              height=plots_height,
                              tag="plot_best_length"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(
                        dpg.mvXAxis, label="iteration", tag="x_axis_best_length")
                    dpg.add_plot_axis(
                        dpg.mvYAxis, label="best length", tag="y_axis_best_length")
                    # dpg.set_axis_limits("y_axis_temp", 0, data_cb["init_temp"])
                    # dpg.add_line_series(
                    #     temperature_x, temperature_y, parent="y_axis_temp",
                    #     tag="series_tag_temp")
                # dpg.bind_item_theme("series_tag_temp", "plot_theme_temp")
                # with dpg.plot(label="Best Energy",
                #               width=plots_width,
                #               height=plots_height,
                #               tag="plot_energy"):
                #     dpg.add_plot_legend()
                #     dpg.add_plot_axis(
                #         dpg.mvXAxis, label="time, s", tag="x_axis_energy")
                #     dpg.add_plot_axis(
                #         dpg.mvYAxis, label="energy", tag="y_axis_energy")
                #     # dpg.add_line_series(
                #     #     best_energy_x, best_energy_y, parent="y_axis_energy",
                #     #     tag="series_tag_energy")
                # # dpg.bind_item_theme("series_tag_energy", "plot_theme_energy")
                # with dpg.plot(label="Bad Solutions",
                #               width=plots_width,
                #               height=plots_height,
                #               tag="plot_bads"):
                #     dpg.add_plot_legend()
                #     dpg.add_plot_axis(
                #         dpg.mvXAxis, label="time, s", tag="x_axis_bad")
                #     dpg.add_plot_axis(
                #         dpg.mvYAxis, label="bad solutions", tag="y_axis_bad")
                #     # dpg.add_line_series(
                #     #     bad_solutions_x, bad_solutions_y, parent="y_axis_bad",
                #     #     tag="series_tag_bad")
                # # dpg.bind_item_theme("series_tag_bad", "plot_theme_bads")

    update_layout()

    dpg.setup_dearpygui()
    dpg.set_primary_window("fullscreen", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    app()
