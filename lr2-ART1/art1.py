#!/usr/bin/env python3
# Python3.12

import random
import argparse
import time

import pandas as pd
import colorama

CC = colorama.Fore.CYAN
CG = colorama.Fore.GREEN
CY = colorama.Fore.YELLOW
CR = colorama.Fore.RED
C0 = colorama.Style.RESET_ALL

# Maximum number of items in a vector
MAX_ITEMS = 8  # WILL BE OVERWRITTEN
# Maximum number of customers
MAX_CUSTOMERS = 500  # WILL BE OVERWRITTEN
TOTAL_PROTOTYPE_VECTORS = 5  # WILL BE OVERWRITTEN

# Beta is a user-defined parameter that controls the degree of overlap
beta = 1.0  # WILL BE OVERWRITTEN
# Vigilance is a user-defined parameter that controls the degree of
# similarity between a prototype vector and an input vector
vigilance = 0.9  # WILL BE OVERWRITTEN

# Number of prototype vectors created so far
num_prototype_vectors = 0

prototype_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
sum_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
members = [0] * TOTAL_PROTOTYPE_VECTORS
membership = [-1] * MAX_CUSTOMERS


def initialize(file, shuffle=False):
    global prototype_vector, sum_vector, members, membership, data, \
        item_name, database, MAX_ITEMS

    # read file data/bank_churn_dataset/train.csv
    data = pd.read_csv(file)

    # if filename is bank_churn_dataset/train.csv
    # then we need to preprocess data
    if "bank_churn_dataset/train.csv" in file.replace("\\", "/"):
        data['Balance'] = data['Balance'].apply(lambda x: 0 if x == 0 else 1)
        data['NumOfProducts'] = data['NumOfProducts'].apply(
            lambda x: 0 if x == 1 else 1)
        data['IsActiveMember'] = data['IsActiveMember'].apply(
            lambda x: 0 if x == 0 else 1)
        data['Age_bin'] = data['Age_bin'].apply(lambda x: 0 if x < 2 else 1)
        data.drop(['Exited'], axis=1, inplace=True)
    elif "bank_churn_dataset/test.csv" in file.replace("\\", "/"):
        data['Balance'] = data['Balance'].apply(lambda x: 0 if x == 0 else 1)
        data['NumOfProducts'] = data['NumOfProducts'].apply(
            lambda x: 0 if x == 1 else 1)
        data['IsActiveMember'] = data['IsActiveMember'].apply(
            lambda x: 0 if x == 0 else 1)
        data['Age_bin'] = data['Age_bin'].apply(lambda x: 0 if x < 2 else 1)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    item_name = data.columns.tolist()
    database = data.values.tolist()

    # check if database has enough columns to fullfill MAX_ITEMS
    if MAX_ITEMS is None:
        MAX_ITEMS = len(item_name)
    elif len(item_name) < MAX_ITEMS:
        raise AssertionError(f"{CR}Database has only {len(item_name)} columns, "
                             f"but {MAX_ITEMS} are required{C0}")
    if MAX_CUSTOMERS > len(database):
        raise AssertionError(f"{CR}Database has only {len(database)} lines, "
                             f"but {MAX_CUSTOMERS} are required{C0}")

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
    """Create new prototype vector from example

    Args:
        example (list): list of items

    Raises:
        AssertionError: No available cluster

    Returns:
        int: cluster
    """
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


time_to_calculate = 0


def perform_clustering():
    global membership, members, num_prototype_vectors, time_to_calculate
    start = time.time()
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

    time_to_calculate = time.time() - start


def display_clusters():
    print(f"\n{CC}Total Members{C0}: {MAX_CUSTOMERS}")
    print(f"{CC}Total Prototype Vectors{C0}: {TOTAL_PROTOTYPE_VECTORS}")
    print(f"{CC}Num Prototype Vectors{C0}: {num_prototype_vectors}")
    print(f"{CC}Vector Size{C0}: {MAX_ITEMS}")
    print(f"{CC}Vigilance{C0}: {vigilance}")
    print(f"{CC}Beta{C0}: {beta}")
    print(f"{CC}Database size{C0}: {len(database)} lines")
    print(f"{CC}Database HEAD{C0}:")
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

    i = 0
    for cluster, members in clusters.items():
        str_PV = ''
        if len(prototype_vector[cluster]) > 50:
            # convert all values to hex string as bits
            str_PV = ''.join([f"{x:01d}" for x in prototype_vector[cluster]])
            str_PV = f'{hex(int(str_PV, 2))}'
        else:
            str_PV = f"{prototype_vector[cluster]}"

        print()
        print(f"{CC}Cluster{C0} {i + 1}, {CC}PV{
              C0} ({len(prototype_vector[cluster])}){str_PV}:")
        i += 1
        limit = 200
        print(f" \\ {CG}Members{C0}: ({len(members)}){(
            members if len(members) < limit else
            f" {CY}>{limit}, not displaying{C0}") if members else "None"}")

    print(f"\n{CC}Time to calculate{C0}: {time_to_calculate:.6f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--beta", type=float,
                        help="Set beta value")
    parser.add_argument("-v", "--vigilance", type=float,
                        help="Set vigilance value")
    parser.add_argument("-m", "--max-customers", type=int,
                        help="Set max customers")
    parser.add_argument("-i", "--max-items", type=int, default=None,
                        help="Set max items, if None, all items will be used")
    parser.add_argument("-p", "--total-prototype-vectors", type=int,
                        help="Set total prototype vectors")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Set file path to read data from")
    parser.add_argument("-s", "--shuffle", action="store_true",
                        help="Shuffle data")
    args = parser.parse_args()

    if args.beta is None or args.vigilance is None \
        or args.max_customers is None or args.total_prototype_vectors is None \
            or args.file is None:
        parser.print_help()
        return

    if args.max_customers < 1:
        raise AssertionError("Max customers must be greater than 0")
    if args.total_prototype_vectors < 1:
        raise AssertionError("Total prototype vectors must be greater than 0")

    global beta, vigilance, MAX_CUSTOMERS, MAX_ITEMS, TOTAL_PROTOTYPE_VECTORS
    beta = args.beta
    vigilance = args.vigilance
    MAX_CUSTOMERS = args.max_customers
    MAX_ITEMS = args.max_items
    TOTAL_PROTOTYPE_VECTORS = args.total_prototype_vectors

    random.seed()
    initialize(args.file, args.shuffle)
    perform_clustering()
    display_clusters()


if __name__ == "__main__":
    main()
