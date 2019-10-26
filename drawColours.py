import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy


# Reads the file  of colours
# Returns the number of colours in the file and a list with the colours (RGB) values

def read_file(fname):
    with open(fname, 'r') as afile:
        lines = afile.readlines()
    n = int(lines[3])  # number of colours  in the file
    col = []
    lines = lines[4:]  # colors as rgb values
    for l in lines:
        rgb = l.split()
        col.append(rgb)
    return n, col


# Display the colours in the order of the permutation in a pyplot window
# Input, list of colours, and ordering  of colours.
# They need to be of the same length

def plot_colours(col, perm):
    assert len(col) == len(perm)

    ratio = 100  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(col), 3))
    for i in range(0, len(col)):
        img[:, i, :] = colours[perm[i]]

    fig, axes = plt.subplots(1, figsize=(8, 4))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


def calculate_distance(colour1, colour2):
    d = math.sqrt(((float(colour2[0]) - float(colour1[0])) ** 2) + ((float(colour2[1]) - float(colour1[1])) ** 2)
                  + ((float(colour2[2]) - float(colour1[2])) ** 2))
    return d


def evaluate(solution):
    total_distance = 0

    for i in range(0, len(solution[0]) - 1):
        total_distance += calculate_distance(solution[0][i], solution[0][i + 1])

    return total_distance


def local_optima(solution):
    original = copy.deepcopy(solution)

    original_distance = evaluate(original)

    neighbour = copy.deepcopy(solution)

    is_optima = True

    for i in range(0, (len(solution[0]) - 1)):

        # assign temp vars to hold array contents to perform switch
        temp_first = copy.deepcopy(neighbour[0][i])
        temp_second = copy.deepcopy(neighbour[0][i + 1])

        # switch values in array
        neighbour[0][i] = temp_second
        neighbour[0][i + 1] = temp_first

        # if the created neighbour has less distance than the original solution, the local optima has not been found
        if evaluate(neighbour) < original_distance:
            is_optima = False
            break

        neighbour = copy.deepcopy(original)

    return is_optima


def random_neighbour(solution):
    neighbour = copy.deepcopy(solution)

    position_to_flip = 0
    position_to_flip2 = 0

    while position_to_flip == position_to_flip2:
        position_to_flip = random.randint(0, len(solution[0]) - 2)
        position_to_flip2 = random.randint(0, len(solution[0]) - 2)

    # IS COPY NEEDED FOR THE NEXT 4 LINES?*******************************************************************************
    temp_first_col = copy.deepcopy(neighbour[0][position_to_flip])
    temp_second_col = copy.deepcopy(neighbour[0][position_to_flip2])

    temp_first_perm = copy.deepcopy(neighbour[1][position_to_flip])
    temp_second_perm = copy.deepcopy(neighbour[1][position_to_flip2])

    neighbour[0][position_to_flip] = temp_second_col
    neighbour[0][position_to_flip2] = temp_first_col

    neighbour[1][position_to_flip] = temp_second_perm
    neighbour[1][position_to_flip2] = temp_first_perm

    return neighbour


#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file4---------------------------------

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 1000  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

permutation = random.sample(range(test_size),
                            test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

plot_colours(test_colours, permutation)


def random_hill_climbing(num):
    random_cols = []

    # permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
    permutation = random.sample(range(test_size),
                                test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

    for i in range(len(test_colours)):
        random_cols.append(test_colours[permutation[i]])

    random_sol = [random_cols, permutation]
    print("Initial Permutation:", random_sol[1])

    print("Initial:", random_sol)

    best_distance = evaluate(random_sol)
    print("Initial Distance:", best_distance)

    sol = copy.deepcopy(random_sol)

    # while not local_optima(sol):

    k = 0

    while k < num:

        # This was changing sol
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


def multi_hill_climb(iter):
    sol = []
    k = 0
    best_sol = []

    for i in range(iter):
        sol.append(random_hill_climbing(1500))

    k = evaluate(sol[0])

    for j in range(len(sol)):

        print(sol[j])
        if evaluate(sol[j]) < k:
            k = evaluate(sol[j])
            best_sol = copy.deepcopy(sol[j])

    plot_colours(test_colours, best_sol[1])

    print(best_sol)

    for l in range(len(sol)):
        print(evaluate(sol[l]))

    print("Best Sol:", evaluate(best_sol))


# solution2 = random_hill_climbing(5000)

# print("Final Permutation:", solution2[1])
# print("Final:", solution2)
# print("Final Distance:", evaluate(solution2))

# plot_colours(test_colours, solution2[1])

# multi_hill_climb(20)

a = [1, 2, 9, 6, 5, 3, 1, 4, 6, 7]
i = 2  # reverse from this index position
j = 6  # reverse until this index position (included)
a[i:j + 1] = reversed(a[i:j + 1])
print(a)  # [1, 2, 5, 6, 9]
