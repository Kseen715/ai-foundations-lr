#!/usr/bin/env python3
# Python3.12

import random
import math

import pandas as pd
import colorama

CC = colorama.Fore.CYAN
CG = colorama.Fore.GREEN
CY = colorama.Fore.YELLOW
CR = colorama.Style.RESET_ALL

MAX_ITEMS = 8
MAX_CUSTOMERS = 5000
TOTAL_PROTOTYPE_VECTORS = 10

beta = 1.0
vigilance = 0.9

num_prototype_vectors = 0

prototype_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
sum_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
members = [0] * TOTAL_PROTOTYPE_VECTORS
membership = [-1] * MAX_CUSTOMERS

# Отток банковских клиентов - покинул банк / остался, основываясь на
# различных признаках

# read file data/bank_churn_dataset/train.csv
data = pd.read_csv('data/bank_churn_dataset/test.csv')

data['Balance'] = data['Balance'].\
    apply(lambda x: 0 if x == 0 else 1)
data['NumOfProducts'] = data['NumOfProducts'].\
    apply(lambda x: 0 if x == 1 else 1)
data['IsActiveMember'] = data['IsActiveMember'].\
    apply(lambda x: 0 if x == 0 else 1)
data['Age_bin'] = data['Age_bin'].\
    apply(lambda x: 0 if x < 2 else 1)


item_name = data.columns.tolist()


database = data.values.tolist()


def initialize():
    global prototype_vector, sum_vector, members, membership
    prototype_vector = [
        [0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
    sum_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
    members = [0] * TOTAL_PROTOTYPE_VECTORS
    membership = [-1] * MAX_CUSTOMERS


def vector_magnitude(vector):
    return sum(vector)


def vector_bitwise_and(v, w):
    result = []
    for i in range(MAX_ITEMS):
        result.append(v[i] and w[i])
    return result


def create_new_prototype_vector(example):
    global num_prototype_vectors
    for cluster in range(TOTAL_PROTOTYPE_VECTORS):
        if members[cluster] == 0:
            break
    if cluster == TOTAL_PROTOTYPE_VECTORS:
        raise AssertionError("No available cluster")
    num_prototype_vectors += 1
    for i in range(MAX_ITEMS):
        prototype_vector[cluster][i] = example[i]
    members[cluster] = 1
    return cluster


def update_prototype_vectors(cluster):
    assert cluster >= 0
    for item in range(MAX_ITEMS):
        prototype_vector[cluster][item] = 0
        sum_vector[cluster][item] = 0
    first = True
    for customer in range(MAX_CUSTOMERS):
        if membership[customer] == cluster:
            if first:
                for item in range(MAX_ITEMS):
                    prototype_vector[cluster][item] = database[customer][item]
                    sum_vector[cluster][item] = database[customer][item]
                first = False
            else:
                for item in range(MAX_ITEMS):
                    prototype_vector[cluster][item] = \
                        prototype_vector[cluster][item] \
                        and database[customer][item]
                    sum_vector[cluster][item] += database[customer][item]


def perform_clustering():
    global membership, members, num_prototype_vectors
    andresult = [0] * MAX_ITEMS
    done = False
    count = 50  # Maximum number of iterations to prevent infinite loops

    while not done:
        done = True
        for index in range(MAX_CUSTOMERS):
            best_cluster = -1
            best_similarity = -1

            for pvec in range(TOTAL_PROTOTYPE_VECTORS):
                if members[pvec]:
                    andresult = vector_bitwise_and(
                        database[index], prototype_vector[pvec])
                    magPE = vector_magnitude(andresult)
                    magP = vector_magnitude(prototype_vector[pvec])
                    magE = vector_magnitude(database[index])
                    similarity = magPE / (beta + magP)
                    threshold = magE / (beta + MAX_ITEMS)

                    if similarity > threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = pvec

            if best_cluster != -1 and (membership[index] != best_cluster):
                old_cluster = membership[index]
                membership[index] = best_cluster
                if old_cluster >= 0:
                    members[old_cluster] -= 1
                    if members[old_cluster] == 0:
                        num_prototype_vectors -= 1
                members[best_cluster] += 1
                if old_cluster >= 0 and old_cluster < TOTAL_PROTOTYPE_VECTORS:
                    update_prototype_vectors(old_cluster)
                update_prototype_vectors(best_cluster)
                done = False

            if membership[index] == -1:
                membership[index] = create_new_prototype_vector(
                    database[index])
                done = False

        if not count:
            break
        count -= 1


def display_clusters():
    print(f"\n{CC}Total Members{CR}: {MAX_CUSTOMERS}")
    print(f"{CC}Total Prototype Vectors{CR}: {num_prototype_vectors}")
    print(f"{CC}Total Clusters{CR}: {TOTAL_PROTOTYPE_VECTORS}")
    print(f"{CC}Vigilance{CR}: {vigilance}")
    print(f"{CC}Beta{CR}: {beta}")
    print(f"{CC}Database size{CR}: {len(database)} lines")
    print(f"{CC}Database head{CR}:")
    print(data.head())

    clusters = {}
    for customer in range(MAX_CUSTOMERS):
        cluster = membership[customer]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(customer)

    # sort clusters by members count
    clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]),
                           reverse=True))

    for cluster, members in clusters.items():
        print()
        print(f"{CC}Cluster{CR} {cluster}, {CC}PV{
              CR} {prototype_vector[cluster]}:")
        print(f" \\ {CG}Members{CR}: ({len(members)}){(
            members if len(members) < 200 else
            f" {CY}>200, not displaying{CR}") if members else "None"}")


def main():
    random.seed()
    initialize()
    perform_clustering()
    display_clusters()


if __name__ == "__main__":
    main()
