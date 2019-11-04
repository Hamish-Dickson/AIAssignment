import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy

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


# calculates distance between 2 colours using euclidean distance formula
def calculate_distance(colour1, colour2):
    return sqrt(((colour2[0] - colour1[0]) ** 2) + ((colour2[1] - colour1[1]) ** 2)
                + ((colour2[2] - colour1[2]) ** 2))


# return the total distance between colours in a given solution
def evaluate(solution):
    total_distance = 0

    for i in range(0, len(solution[0]) - 1):
        total_distance += calculate_distance(solution[0][i], solution[0][i + 1])

    return total_distance


def generate_random_solution():
    # Get the directory where the file is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)  # Change the working directory so we can read the file4---------------------------------

    ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

    test_size = 500  # Size of the subset of colours for testing
    test_colours = colours[0:test_size]  # list of colours for testing

    # permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
    permutation = random.sample(range(test_size),
                                test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

    random_cols = []

    # Creates and plots a random solution
    for i in range(len(test_colours)):
        random_cols.append(test_colours[permutation[i]])

    random_sol = [random_cols, permutation]

    return random_sol


# Iterates through each colour and finds the proceeding colour with the shortest distance between them and swaps it with
# the colour next to itself
# Need to rearrange the colours to allow sorting, then revert back to the beginning solution but using the newly
# ordered permutation
def solve(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    # The index where the current closest colour is stored
    temp_space = 0

    for j in range(len(sorted_solution[0]) - 1):

        smaller_found = False
        # Finds the first distance to use as the starting point
        smallest_dist = calculate_distance(sol[0][j], sol[0][j + 1])

        for k in range(j + 2, len(sorted_solution[0])):

            dist = calculate_distance(sol[0][j], sol[0][k])

            if dist < smallest_dist:
                smallest_dist = dist
                smaller_found = True

                temp_space = k
                temp_col2 = sol[0][k]
                temp_perm2 = sol[1][k]

        if smaller_found:
            temp_col1 = sol[0][j + 1]
            temp_perm1 = sol[1][j + 1]

            sol[0][j + 1] = temp_col2
            sol[0][temp_space] = temp_col1

            sol[1][j + 1] = temp_perm2
            sol[1][temp_space] = temp_perm1

    sorted_solution[1] = sol[1]

    return sorted_solution


# experimental function to find if the provided solution is the local optima solution
def local_optima(solution):
    original = copy.deepcopy(solution)  # assign and keep the original solution for later use

    original_distance = evaluate(original)  # evaluate the original solution

    neighbour = copy.deepcopy(solution)  # create a copy of the current solution that can be mutated
    better_solution = copy.deepcopy(solution)  # variable to hold the best variable found in neighbourhood if any

    is_optima = True

    for i in range(0, len(solution[0]) - 1):
        # assign temp vars to hold array contents to perform switch
        temp_first = copy.deepcopy(neighbour[0][i])
        temp_second = copy.deepcopy(neighbour[0][i + 1])
        temp_first_perm = copy.deepcopy(neighbour[1][i])
        temp_second_perm = copy.deepcopy(neighbour[1][i + 1])

        # switch values in array
        neighbour[0][i] = temp_second
        neighbour[0][i + 1] = temp_first
        neighbour[1][i] = temp_second_perm
        neighbour[1][i + 1] = temp_first_perm

        # if the created neighbour has less distance than the original solution, the local optima has not been found
        if evaluate(neighbour) < original_distance:
            is_optima = False
            better_solution = neighbour
            break

        neighbour = copy.deepcopy(original)

    return is_optima, better_solution


# method to generate a random neighbouring solution
def random_neighbour(solution):
    neighbour = copy.deepcopy(solution)

    first_position_to_flip = random.randint(0, len(solution[0]) - 2)
    second_position_to_flip = random.randint(0, len(solution[0]) - 2)

    # assign temporary variables to hold for the flip operation
    temp_first = neighbour[0][first_position_to_flip].copy()
    temp_second = neighbour[0][second_position_to_flip].copy()
    temp_permutation_first = neighbour[1][first_position_to_flip]
    temp_permutation_second = neighbour[1][second_position_to_flip]

    # perform the flip operations
    neighbour[0][first_position_to_flip] = temp_second
    neighbour[0][second_position_to_flip] = temp_first
    neighbour[1][first_position_to_flip] = temp_permutation_second
    neighbour[1][second_position_to_flip] = temp_permutation_first

    return neighbour


# Calculates a random neighbour by reversing a random section of the list.
def random_neighbour_ryan(solution):
    neighbour = copy.deepcopy(solution)

    position_to_flip = 0
    position_to_flip2 = 0

    while position_to_flip == position_to_flip2:
        position_to_flip = random.randint(0, len(solution[0]) - 2)
        position_to_flip2 = random.randint(0, len(solution[0]) - 2)

    # If the first random number is less than the second random number
    if position_to_flip < position_to_flip2:
        neighbour[1][position_to_flip:position_to_flip2 + 1] = reversed(
            neighbour[1][position_to_flip:position_to_flip2 + 1])
        # If the second random number is less that the first random number
    else:
        neighbour[1][position_to_flip2:position_to_flip + 1] = reversed(
            neighbour[1][position_to_flip2:position_to_flip + 1])

    return neighbour


# plot_colours(test_colours, permutation)

# method to perform a single random hill climb
def random_hill_climbing_local_optima():
    random_cols = []

    permutation = random.sample(range(test_size),
                                test_size)

    for i in range(len(test_colours)):
        random_cols.append(test_colours[permutation[i]])  # build the initial random solution

    random_sol = [random_cols, permutation]
    best_distance = evaluate(random_sol)  # calculate the distance of the initial solution
    # print("initial: ", best_distance)

    sol = copy.deepcopy(random_sol)

    while True:
        is_optima, sol = local_optima(sol)  # get if the current solution is the optima, as well as any improved
        # solution found
        if is_optima:  # if the local optima has been found, exit the loop
            break
        else:
            # create a new random neighbouring solution to test
            neighbour = random_neighbour(sol)

            # evaluate the newly created neighbour
            neighbour_distance = evaluate(neighbour)

            # if the neighbouring solution has a better distance than the current best distance
            if neighbour_distance < best_distance:
                sol = neighbour
                best_distance = neighbour_distance
                # print("new neighbour: ", best_distance)

    return sol, evaluate(sol)


def random_hill_climbing(num):
    random_sol = generate_random_solution()

    print("Initial Permutation:", random_sol[1])

    print("Initial:", random_sol)

    best_distance = evaluate(random_sol)
    print("Initial Distance:", best_distance)

    sol = solve(random_sol)

    k = 0

    while k < num:

        neighbour = random_neighbour(sol)

        neighbour_distance = evaluate(neighbour)

        if neighbour_distance < best_distance:
            sol = copy.deepcopy(neighbour)

            best_distance = neighbour_distance

            # print("New Permutation:", sol[1])
            # print("New Distance:", evaluate(sol))
            # print("New Sol:", sol)

        k += 1

    return sol


# method to perform multiple hill climbs, taking in the number of tries to be attempted
def multi_hill_climb(tries):
    distances = []
    best_sol = []
    best_distance = 9999999999  # random high value, since we are trying to find a minimum


    for i in range(0, tries):
        hill_climb = random_hill_climbing_local_optima()
        distances.append(hill_climb[1])
        if hill_climb[1] < best_distance:
            best_distance = hill_climb[1]
            best_sol = hill_climb[0]

    print("from", tries, "Multi-HC found", "\nthe best distance of: ", best_distance,
          "\nusing solution: ", best_sol)
    plot_colours(best_sol[0], best_sol[1])


def multi_hill_climb_ryan(iter):
    sol = []

    for i in range(iter):
        sol.append(random_hill_climbing(50))

    best_sol = sol[0]
    k = evaluate(sol[0])

    for j in range(len(sol)):

        print(sol[j])
        if evaluate(sol[j]) < k:
            k = evaluate(sol[j])
            best_sol = copy.deepcopy(sol[j])

    print(best_sol)

    print("Best Sol:", evaluate(best_sol))

    return best_sol


def organise_avg(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for j in range(len(sol[0])):
        avg_col[j] = sum(sol[0][j]) / len(sol[0][j])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l] > avg_col[l + 1]:
                temp = avg_col[l]

                avg_col[l] = avg_col[l + 1]
                avg_col[l + 1] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_red(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l][0] < avg_col[l + 1][0]:
                temp = avg_col[l][0]

                avg_col[l][0] = avg_col[l + 1][0]
                avg_col[l + 1][0] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_green(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l][1] < avg_col[l + 1][1]:
                temp = avg_col[l][1]

                avg_col[l][1] = avg_col[l + 1][1]
                avg_col[l + 1][1] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_blue(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l][2] < avg_col[l + 1][2]:
                temp = avg_col[l][2]

                avg_col[l][2] = avg_col[l + 1][2]
                avg_col[l + 1][2] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_ratio_red(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for j in range(len(sol[0])):
        avg_col[j] = sol[0][j][0] / (sol[0][j][1] + sol[0][j][2])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l] < avg_col[l + 1]:
                temp = avg_col[l]

                avg_col[l] = avg_col[l + 1]
                avg_col[l + 1] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_ratio_green(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for j in range(len(sol[0])):
        avg_col[j] = sol[0][j][1] / (sol[0][j][0] + sol[0][j][2])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l] < avg_col[l + 1]:
                temp = avg_col[l]

                avg_col[l] = avg_col[l + 1]
                avg_col[l + 1] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


def organise_ratio_blue(start_solution):
    sorted_solution = copy.deepcopy(start_solution)

    sol = arrange_colours(sorted_solution)

    avg_col = copy.deepcopy(sol[0])

    for j in range(len(sol[0])):
        avg_col[j] = sol[0][j][2] / (sol[0][j][0] + sol[0][j][1])

    for k in range(len(sol[0])):

        for l in range(len(sol[0]) - k - 1):

            if avg_col[l] < avg_col[l + 1]:
                temp = avg_col[l]

                avg_col[l] = avg_col[l + 1]
                avg_col[l + 1] = temp

                temp_perm = sol[1][l]

                sol[1][l] = sol[1][l + 1]
                sol[1][l + 1] = temp_perm

    sorted_solution[1] = sol[1]

    return sorted_solution


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

multi_hill_climb(100)
multi_hill_climb(1000)
multi_hill_climb(5000)
multi_hill_climb(25000)

exit()