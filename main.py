import dearpygui.dearpygui as dpg

import random
import math

MAX_LENGTH = 10
INITIAL_TEMPERATURE = 100.0
FINAL_TEMPERATURE = 0.5
ALPHA = 0.99
STEPS_PER_CHANGE = 1000

class MemberType:
    def __init__(self):
        self.solution = list(range(MAX_LENGTH))
        self.energy = 0.0

def get_rand(max_length):
    return random.randint(0, max_length - 1)

def get_srand():
    return random.random()

def tweak_solution(member):
    x = get_rand(MAX_LENGTH)
    y = get_rand(MAX_LENGTH)
    while x == y:
        y = get_rand(MAX_LENGTH)
    member.solution[x], member.solution[y] = member.solution[y], member.solution[x]

def initialize_solution(member):
    random.shuffle(member.solution)
    for _ in range(MAX_LENGTH):
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
    timer = 0
    temperature = INITIAL_TEMPERATURE
    current = MemberType()
    working = MemberType()
    best = MemberType()
    best.energy = 100.0

    initialize_solution(current)
    compute_energy(current)
    copy_solution(working, current)

    while temperature > FINAL_TEMPERATURE:
        print(f"Temperature : {temperature}")
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
                copy_solution(working, current)
        print(f"Best energy = {best.energy}")
        temperature *= ALPHA

    emit_solution(best)


if __name__ == "__main__":
    main()
    exit(1)
    dpg.create_context()
    dpg.create_viewport(title='N-Queen Task', width=600, height=300)

    with dpg.window(label="Example Window", tag="fullscreen"):
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
        dpg.add_button(label="run")

    dpg.setup_dearpygui()
    dpg.set_primary_window("fullscreen", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()