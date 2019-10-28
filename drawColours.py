import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from math import sqrt


# Reads the file  of colours
# Returns the number of colours in the file and a list with the colours (RGB) values

def read_file(fname):
    with open(fname, 'r') as afile:
        lines = afile.readlines()
    n = int(lines[3])  # number of colours  in the file
    col = []
    lines = lines[4:]  # colors as rgb values
    rgb = []
    for l in lines:
        rgb = l.split()
        for i in range(0, len(rgb)):
            rgb[i] = float(rgb[i])

        col.append(rgb)
    return n, col


# Display the colours in the order of the permutation in a pyplot window
# Input, list of colours, and ordering  of colours.
# They need to be of the same length

def plot_colours(col, perm):
    assert len(col) == len(perm)

    ratio = 10  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(col), 3))
    for i in range(0, len(col)):
        img[:, i, :] = colours[perm[i]]

    fig, axes = plt.subplots(1, figsize=(8, 4))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


def calculate_distance(colour1, colour2):
    #return sqrt(((float(colour2[0]) - float(colour1[0])) ** 2) + ((float(colour2[1]) - float(colour1[1])) ** 2)
     #           + ((float(colour2[2]) - float(colour1[2])) ** 2))

    return sqrt(((colour2[0] - colour1[0]) ** 2) + ((colour2[1] - colour1[1]) ** 2)
                + ((colour2[2] - colour1[2]) ** 2))


# return math.sqrt(((float(colour2[0]) - float(colour1[0])) * (float(colour2[0]) - float(colour1[0])))
#                + ((float(colour2[1]) - float(colour1[1])) * (float(colour2[1]) - float(colour1[1])))
#               + ((float(colour2[2]) - float(colour1[2])) * (float(colour2[2]) - float(colour1[2]))))


def evaluate(solution):
    total_distance = 0

    for i in range(0, len(solution) - 1):
        total_distance += calculate_distance(solution[i], solution[i + 1])

    return total_distance


def local_optima(solution):
    original = solution.copy()

    original_distance = evaluate(original)

    neighbour = solution.copy()
    better_solution = solution.copy()

    is_optima = True

    for i in range(0, len(solution) - 1):
        # assign temp vars to hold array contents to perform switch
        temp_first = neighbour[i].copy()
        temp_second = neighbour[i + 1].copy()

        # switch values in array
        neighbour[i] = temp_second
        neighbour[i + 1] = temp_first

        # if the created neighbour has less distance than the original solution, the local optima has not been found
        if evaluate(neighbour) < original_distance:
            is_optima = False
            better_solution = neighbour
            break

        neighbour = original.copy()

    return is_optima, better_solution


def random_neighbour(solution, permutation):
    neighbour = solution.copy()
    perm = permutation.copy()

    first_position_to_flip = random.randint(0, len(solution) - 2)
    second_position_to_flip = random.randint(0, len(solution) - 2)

    temp_first = neighbour[first_position_to_flip].copy()
    temp_second = neighbour[second_position_to_flip].copy()
    temp_permutaion_first = permutation[first_position_to_flip]
    temp_permutaion_second = permutation[second_position_to_flip]

    neighbour[first_position_to_flip] = temp_second
    neighbour[second_position_to_flip] = temp_first
    perm[first_position_to_flip] = temp_permutaion_second
    perm[second_position_to_flip] = temp_permutaion_first

    return neighbour, perm


#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 20  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

# permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
permutation = random.sample(range(test_size),
                            test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1


# plot_colours(test_colours, permutation)


def random_hill_climbing():
    random_sol = []

    permutation = random.sample(range(test_size),
                                test_size)

    for i in range(len(test_colours)):
        random_sol.append(test_colours[permutation[i]])

    best_distance = evaluate(random_sol)
    # print("initial: ", best_distance)

    sol = random_sol.copy()
    curr_permutation = permutation.copy()

    while True:
        is_optima, sol = local_optima(sol)
        if is_optima:
            break
        else:
            neighbour, curr_permutation = random_neighbour(sol, curr_permutation.copy())

            neighbour_distance = evaluate(neighbour)

            if neighbour_distance < best_distance:
                sol = neighbour
                best_distance = neighbour_distance
                # print("new neighbour: ", best_distance)

        # while not local_optima(sol):

    return sol, curr_permutation, evaluate(sol)


def multi_hill_climb(tries):
    distances = []
    best_sol = []
    best_distance = 9999
    best_permutation = []

    for i in range(0, tries):
        hill_climb = random_hill_climbing()
        distances.append(hill_climb[2])
        if hill_climb[2] < best_distance:
            best_distance = hill_climb[2]
            best_sol = hill_climb[0]
            best_permutation = hill_climb[1]

    print("from", tries, "Multi-HC found", "\nthe best distance of: ", best_distance,
          "\nusing solution: ", best_sol, "using permutation: ", best_permutation)
    plot_colours(best_sol, best_permutation)


multi_hill_climb(100)
multi_hill_climb(1000)
multi_hill_climb(5000)
multi_hill_climb(25000)
