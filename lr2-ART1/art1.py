#!/usr/bin/env python3
# Python3.12

import random
import math
import pandas as pd

MAX_ITEMS = 11
MAX_CUSTOMERS = 10
TOTAL_PROTOTYPE_VECTORS = 5

beta = 1.0
vigilance = 0.9

num_prototype_vectors = 0

prototype_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
sum_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
members = [0] * TOTAL_PROTOTYPE_VECTORS
membership = [-1] * MAX_CUSTOMERS

item_name = [
    "Hammer", "Paper", "Snickers", "Screwdriver", 
    "Pen", "Kit-Kat", "Wrench", "Pencil", 
    "Heath-Bar", "Tape-Measure", "Binder"
]

database = [
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
]

def display_customer_database():
    print("\n")
    for cluster in range(TOTAL_PROTOTYPE_VECTORS):
        print(f"ProtoVector {cluster:2d} : ", end="")
        for item in range(MAX_ITEMS):
            print(f"{prototype_vector[cluster][item]:1d} ", end="")
        print("\n\n")
        for customer in range(MAX_CUSTOMERS):
            if membership[customer] == cluster:
                print(f"Customer {customer:2d}    : ", end="")
                for item in range(MAX_ITEMS):
                    print(f"{database[customer][item]:1d} ", end="")
                print(f"  : {membership[customer]} : \n")
        print("\n")
    print("\n")

def initialize():
    global prototype_vector, sum_vector, members, membership
    prototype_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
    sum_vector = [[0] * MAX_ITEMS for _ in range(TOTAL_PROTOTYPE_VECTORS)]
    members = [0] * TOTAL_PROTOTYPE_VECTORS
    membership = [-1] * MAX_CUSTOMERS

def vector_magnitude(vector):
    return sum(vector)

def vector_bitwise_and(result, v, w):
    for i in range(MAX_ITEMS):
        result[i] = v[i] and w[i]

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
                    prototype_vector[cluster][item] = prototype_vector[cluster][item] and database[customer][item]
                    sum_vector[cluster][item] += database[customer][item]

def perform_art1():
    global membership, members, num_prototype_vectors
    andresult = [0] * MAX_ITEMS
    done = False
    count = 50
    while not done:
        done = True
        for index in range(MAX_CUSTOMERS):
            for pvec in range(TOTAL_PROTOTYPE_VECTORS):
                if members[pvec]:
                    vector_bitwise_and(andresult, database[index], prototype_vector[pvec])
                    magPE = vector_magnitude(andresult)
                    magP = vector_magnitude(prototype_vector[pvec])
                    magE = vector_magnitude(database[index])
                    result = magPE / (beta + magP)
                    test = magE / (beta + MAX_ITEMS)
                    if result > test:
                        if magPE / magE < vigilance:
                            old = membership[index]
                            if membership[index] != pvec:
                                membership[index] = pvec
                                if old >= 0:
                                    members[old] -= 1
                                    if members[old] == 0:
                                        num_prototype_vectors -= 1
                                members[pvec] += 1
                                if old >= 0 and old < TOTAL_PROTOTYPE_VECTORS:
                                    update_prototype_vectors(old)
                                update_prototype_vectors(pvec)
                                done = False
                                break
            if membership[index] == -1:
                membership[index] = create_new_prototype_vector(database[index])
                done = False
        if not count:
            break
        count -= 1

def make_recommendation(customer):
    best_item = -1
    val = 0
    for item in range(MAX_ITEMS):
        if database[customer][item] == 0 and sum_vector[membership[customer]][item] > val:
            best_item = item
            val = sum_vector[membership[customer]][item]
    print(f"For Customer {customer}, ", end="")
    if best_item >= 0:
        print(f"The best recommendation is {best_item} ({item_name[best_item]})")
        print(f"Owned by {sum_vector[membership[customer]][best_item]} out of {members[membership[customer]]} members of this cluster")
    else:
        print("No recommendation can be made.")
    print("Already owns: ", end="")
    for item in range(MAX_ITEMS):
        if database[customer][item]:
            print(f"{item_name[item]} ", end="")
    print("\n")

def main():
    random.seed()
    initialize()
    perform_art1()
    display_customer_database()
    for customer in range(MAX_CUSTOMERS):
        make_recommendation(customer)

if __name__ == "__main__":
    main()