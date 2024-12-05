import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random


class TravelingSalesmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Задача коммивояжера")

        # Элементы интерфейса
        tk.Label(
            root, text="Размер популяции:"
        ).grid(row=0, column=0, sticky="w")
        self.population_size_entry = tk.Entry(root)
        self.population_size_entry.grid(row=0, column=1)
        self.population_size_entry.insert(0, "100")

        tk.Label(
            root, text="Число поколений:"
        ).grid(row=1, column=0, sticky="w")
        self.generations_entry = tk.Entry(root)
        self.generations_entry.grid(row=1, column=1)
        self.generations_entry.insert(0, "200")

        tk.Label(
            root, text="Вероятность мутации (0-1):"
        ).grid(row=2, column=0, sticky="w")
        self.mutation_rate_entry = tk.Entry(root)
        self.mutation_rate_entry.grid(row=2, column=1)
        self.mutation_rate_entry.insert(0, "0.1")

        self.start_button = tk.Button(
            root, text="Запустить", command=self.run_algorithm
        )
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Поле для визуализации
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

        # Генерация графа
        self.num_nodes = 20
        self.coords = self.generate_graph()

    def generate_graph(self):
        """Генерация случайных координат вершин графа."""
        return np.random.rand(self.num_nodes, 2) * 100

    def distance(self, a, b):
        """Расчет евклидового расстояния между двумя точками."""
        return np.linalg.norm(a - b)

    def fitness(self, route):
        """Функция приспособленности: длина маршрута."""
        return sum(
            self.distance(self.coords[route[i]], self.coords[route[i + 1]])
            for i in range(len(route) - 1)
        ) + self.distance(self.coords[route[-1]], self.coords[route[0]])

    def initialize_population(self, population_size):
        """Инициализация популяции маршрутов."""
        return [
            random.sample(range(self.num_nodes), self.num_nodes)
            for _ in range(population_size)
        ]

    def select_parents(self, population, fitness_values):
        """Рулеточный отбор родителей."""
        total_fitness = sum(1 / fv for fv in fitness_values)
        probabilities = [(1 / fv) / total_fitness for fv in fitness_values]
        return random.choices(population, weights=probabilities, k=2)

    def crossover(self, parent1, parent2):
        """Одноточечное скрещивание."""
        cut = random.randint(1, self.num_nodes - 2)
        child = parent1[:cut] + \
            [gene for gene in parent2 if gene not in parent1[:cut]]
        return child

    def mutate(self, route, mutation_rate):
        """Мутация путем обмена двух случайных генов."""
        if random.random() < mutation_rate:
            i, j = random.sample(range(self.num_nodes), 2)
            route[i], route[j] = route[j], route[i]

    def run_algorithm(self):
        """Запуск генетического алгоритма."""
        try:
            population_size = int(self.population_size_entry.get())
            generations = int(self.generations_entry.get())
            mutation_rate = float(self.mutation_rate_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные параметры.")
            return

        # Инициализация
        population = self.initialize_population(population_size)
        best_route = None
        best_distance = float("inf")
        progress = []

        for generation in range(generations):
            fitness_values = [self.fitness(route) for route in population]
            best_gen_route = population[np.argmin(fitness_values)]
            best_gen_distance = min(fitness_values) # type: ignore

            # Обновление глобального лучшего решения
            if best_gen_distance < best_distance:
                best_distance = best_gen_distance
                best_route = best_gen_route

            # Обновление графика
            self.update_plot(best_route, best_distance, generation)
            progress.append(best_distance)

            # Новое поколение
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = self.select_parents(
                    population, fitness_values)
                child = self.crossover(parent1, parent2)
                self.mutate(child, mutation_rate)
                new_population.append(child)
            population = new_population

        # Финальный результат
        self.update_plot(best_route, best_distance, "Итог")
        messagebox.showinfo("Готово", f"Лучший маршрут: {best_distance:.2f}")

    def update_plot(self, route, distance, generation):
        """Обновление графика."""
        self.ax.clear()
        self.ax.scatter(self.coords[:, 0], self.coords[:, 1], color="red")
        path = [self.coords[city] for city in route] + [self.coords[route[0]]]
        self.ax.plot([p[0] for p in path], [p[1] for p in path], color="blue")
        self.ax.set_title(f"Поколение: {generation}, Длина: {distance:.2f}")
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = TravelingSalesmanApp(root)
    root.mainloop()
