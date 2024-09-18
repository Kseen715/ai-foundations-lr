import dearpygui.dearpygui as dpg

import random
import math
import time
import threading

process = None
is_running = False


MAX_LENGTH = 20
INITIAL_TEMPERATURE = 100.0
FINAL_TEMPERATURE = 0.1
ALPHA = 0.99
STEPS_PER_CHANGE = 100

# Data for plots, x is time when data captured in s, y is value of data
bad_solutions_y = []
bad_solutions_x = []
best_energy_y = []
best_energy_x = []
temperature_y = []
temperature_x = []

start_time = 0



class MemberType:
    def __init__(self):
        self.solution = list(range(MAX_LENGTH))
        self.energy = 0.0

def get_rand(max_length):
    return random.randint(0, max_length - 1)

def get_srand():
    return random.random()

def tweak_solution(member):
    x, y = random.sample(range(MAX_LENGTH), 2)
    member.solution[x], member.solution[y] = member.solution[y], member.solution[x]

def initialize_solution(member):
    random.shuffle(member.solution)
    tweak_solution(member)

def emit_solution(member):
    board = [['.' for _ in range(MAX_LENGTH)] for _ in range(MAX_LENGTH)]
    for x in range(MAX_LENGTH):
        board[x][member.solution[x]] = 'Q'
    for row in board:
        print(' '.join(row))
    print("\n")

def compute_energy(member):
    conflicts = 0
    board = [['.' for _ in range(MAX_LENGTH)] for _ in range(MAX_LENGTH)]
    for i in range(MAX_LENGTH):
        board[i][member.solution[i]] = 'Q'
    dx = [-1, 1, -1, 1]
    dy = [-1, 1, 1, -1]
    for i in range(MAX_LENGTH):
        x, y = i, member.solution[i]
        for j in range(4):
            tempx, tempy = x, y
            while True:
                tempx += dx[j]
                tempy += dy[j]
                if tempx < 0 or tempx >= MAX_LENGTH or tempy < 0 or tempy >= MAX_LENGTH:
                    break
                if board[tempx][tempy] == 'Q':
                    conflicts += 1
    member.energy = float(conflicts)

def copy_solution(dest, src):
    dest.solution = src.solution[:]
    dest.energy = src.energy


def main():
    temperature_x.clear()
    temperature_y.clear()
    best_energy_x.clear()
    best_energy_y.clear()
    bad_solutions_x.clear()
    bad_solutions_y.clear()
    rejected = 0

    temperature = INITIAL_TEMPERATURE
    current = MemberType()
    working = MemberType()
    best = MemberType()
    best.energy = 100.0

    start_time = time.time()

    initialize_solution(current)
    compute_energy(current)
    copy_solution(working, current)
    global is_running
    while (temperature > FINAL_TEMPERATURE):
        if not is_running:
            break
        # print(f"Temperature : {temperature}")
        temperature_x.append(float(time.time() - start_time))
        temperature_y.append(float(temperature))
        accepted = 0
        for _ in range(STEPS_PER_CHANGE):
            tweak_solution(working)
            compute_energy(working)
            delta = working.energy - current.energy
            if delta <= 0 or math.exp(-delta / temperature) > get_srand():
                accepted += 1 if delta > 0 else 0
                copy_solution(current, working)
                if current.energy < best.energy:
                    copy_solution(best, current)
                else:
                    rejected += 1
                    if rejected % 100 == 0:
                        bad_solutions_x.append(float(time.time() - start_time))
                        bad_solutions_y.append(rejected)
            else:
                copy_solution(working, current)
        # print(f"Best energy = {best.energy}")
        best_energy_x.append(float(time.time() - start_time))
        best_energy_y.append(float(best.energy))
        temperature *= ALPHA
        # print(temperature_x)
        # print(temperature_y)
        
        # Fit the axis data
        if len(temperature_x) > 0:
            dpg.set_value("series_tag_temp", [temperature_x, temperature_y])
            # dpg.fit_axis_data("y_axis_temp")
            dpg.fit_axis_data("x_axis_temp")
        if len(best_energy_x) > 0:
            dpg.set_value("series_tag_energy", [best_energy_x, best_energy_y])
            # dpg.fit_axis_data("y_axis_energy")
            dpg.set_axis_limits("y_axis_energy", -0.5, max(best_energy_y) + 0.5)
            dpg.fit_axis_data("x_axis_energy")
        if len(bad_solutions_x) > 0:
            dpg.set_value("series_tag_bad", [bad_solutions_x, bad_solutions_y])
            dpg.fit_axis_data("x_axis_bad")
            dpg.set_axis_limits("y_axis_bad", -0.5, max(bad_solutions_y) + 0.5)



    emit_solution(best)



def run():
    global is_running
    global process
    if process is None:
        is_running = True
        process = threading.Thread(target=main)
        process.start()

def stop():
    global is_running
    global process
    if process is not None:
        is_running = False
        process.join()
        process = None


if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title='N-Queen Task', width=1280, height=720)

    with dpg.window(label="Example Window", tag="fullscreen"):

        with dpg.theme(tag="plot_theme_temp"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 99, 99), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 0, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0, category=dpg.mvThemeCat_Plots)
        with dpg.theme(tag="plot_theme_energy"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (22, 90, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 0, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0, category=dpg.mvThemeCat_Plots)
        with dpg.theme(tag="plot_theme_bads"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (111, 111, 111), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 0, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0, category=dpg.mvThemeCat_Plots)



        dpg.add_text("Hello, world")
        # Максимальная температура
        dpg.add_input_float(label="Max temperature", default_value=100, width=100)
#  Минимальная температура
        dpg.add_input_float(label="Min temperature", default_value=0.01, width=100)
#  Коэффициент понижения температуры
        dpg.add_input_float(label="Temperature reduction coefficient", default_value=0.99, width=100)
#  Количество ферзей.
        dpg.add_input_int(label="Number of queens", default_value=8, width=100)
#  Количество шагов при постоянном значении температуры.
        dpg.add_input_int(label="Number of steps at constant temperature", default_value=10, width=100)
        with dpg.group(horizontal=True):
            dpg.add_button(label="run", callback=run)
            dpg.add_button(label="stop", callback=stop)

        with dpg.group(horizontal=True):
            with dpg.plot(label="Temperature"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time, s", tag="x_axis_temp")
                dpg.add_plot_axis(dpg.mvYAxis, label="temperature", tag="y_axis_temp")
                dpg.set_axis_limits("y_axis_temp", 0, INITIAL_TEMPERATURE)
                dpg.add_line_series(temperature_x, temperature_y, parent="y_axis_temp", tag="series_tag_temp")
            dpg.bind_item_theme("series_tag_temp", "plot_theme_temp")
            with dpg.plot(label="Best Energy"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time, s", tag="x_axis_energy")
                dpg.add_plot_axis(dpg.mvYAxis, label="energy", tag="y_axis_energy")
                dpg.add_line_series(best_energy_x, best_energy_y, parent="y_axis_energy", tag="series_tag_energy")
            dpg.bind_item_theme("series_tag_energy", "plot_theme_energy")
            with dpg.plot(label="Bad Solutions"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time, s", tag="x_axis_bad")
                dpg.add_plot_axis(dpg.mvYAxis, label="bad solutions", tag="y_axis_bad")
                dpg.add_line_series(bad_solutions_x, bad_solutions_y, parent="y_axis_bad", tag="series_tag_bad")
            dpg.bind_item_theme("series_tag_bad", "plot_theme_bads")

    dpg.setup_dearpygui()
    dpg.set_primary_window("fullscreen", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()