import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os


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

    ratio = 10  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
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

    for i in range(0, len(solution)-1):
        total_distance += calculate_distance(solution[i], solution[i+1])

    return total_distance


def local_optima(solution):
    original = solution.copy()

    original_distance = evaluate(original)

    neighbour = solution.copy()

    is_optima = True

    for i in range(0, len(solution)):
        # assign temp vars to hold array contents to perform switch
        temp_first = neighbour[i].copy()
        temp_second = neighbour[i+1].copy()

        # switch values in array
        neighbour[i] = temp_second
        neighbour[i+1] = temp_first

        # if the created neighbour has less distance than the original solution, the local optima has not been found
        if evaluate(neighbour) < original_distance:
            is_optima = False
            break

        neighbour = original.copy()

    return is_optima


#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 1000  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

# permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
permutation = random.sample(range(test_size),
                            test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1
plot_colours(test_colours, permutation)

print(colours)
print(calculate_distance(colours[1], colours[0]))
print(local_optima(colours))
