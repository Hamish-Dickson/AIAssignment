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

    ratio = 100  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
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

    for i in range(0, len(solution) - 1):
        # print(colour_one, colour_two)
        total_distance += calculate_distance(test_colours[solution[i]], test_colours[solution[i + 1]])

    return total_distance


def generate_random_solution():
    # permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
    random_solution = random.sample(range(test_size),
                                    test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

    return random_solution


# Iterates through each colour and finds the proceeding colour with the shortest distance between them and swaps it with
# the colour next to itself
# Need to rearrange the colours to allow sorting, then revert back to the beginning solution but using the newly
# ordered permutation
def solve(start_solution):
    sorted_solution = start_solution.copy()

    # The index where the current closest colour is stored
    temp_space = 0

    for j in range(len(sorted_solution) - 1):

        smaller_found = False
        # Finds the first distance to use as the starting point
        smallest_dist = calculate_distance(test_colours[sorted_solution[j]], test_colours[sorted_solution[j + 1]])

        for k in range(j + 2, len(sorted_solution)):

            dist = calculate_distance(test_colours[sorted_solution[j]], test_colours[sorted_solution[k]])

            if dist < smallest_dist:
                smallest_dist = dist
                smaller_found = True

                temp_space = k
                temp_pos2 = sorted_solution[k]

        if smaller_found:
            temp_pos1 = sorted_solution[j + 1]

            sorted_solution[j + 1] = temp_pos2
            sorted_solution[temp_space] = temp_pos1

    return sorted_solution


# experimental function to find if the provided solution is the local optima solution
def local_optima(solution):
    original = solution.copy()  # assign and keep the original solution for later use

    original_distance = evaluate(original)  # evaluate the original solution

    neighbour = solution.copy()  # create a copy of the current solution that can be mutated
    better_solution = solution.copy()  # variable to hold the best variable found in neighbourhood if any

    is_optima = True

    for i in range(0, len(solution) - 1):
        # assign temp vars to hold array contents to perform switch
        temp_first = neighbour[i]
        temp_second = neighbour[i + 1]

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


# method to generate a random neighbouring solution
def random_neighbour(solution):
    neighbour = solution.copy()

    first_position_to_flip = 0
    second_position_to_flip = 0

    while first_position_to_flip == second_position_to_flip:
        first_position_to_flip = random.randint(0, len(solution) - 2)
        second_position_to_flip = random.randint(0, len(solution) - 2)

    # assign temporary variables to hold for the flip operation
    temp_first = neighbour[first_position_to_flip]
    temp_second = neighbour[second_position_to_flip]

    # perform the flip operations
    neighbour[first_position_to_flip] = temp_second
    neighbour[second_position_to_flip] = temp_first

    return neighbour


# Calculates a random neighbour by reversing a random section of the list.
def random_neighbour_ryan(solution):
    neighbour = solution.copy()

    first_position_to_flip = 0
    second_position_to_flip = 0

    while first_position_to_flip == second_position_to_flip:
        first_position_to_flip = random.randint(0, len(solution) - 2)
        second_position_to_flip = random.randint(0, len(solution) - 2)

    # If the first random number is less than the second random number
    if first_position_to_flip < second_position_to_flip:
        neighbour[first_position_to_flip:second_position_to_flip + 1] = reversed(
            neighbour[first_position_to_flip:second_position_to_flip + 1])
        # If the second random number is less that the first random number
    else:
        neighbour[second_position_to_flip:first_position_to_flip + 1] = reversed(
            neighbour[second_position_to_flip:first_position_to_flip + 1])

    return neighbour

# method to perform a single random hill climb
def random_hill_climbing_local_optima():
    solution = generate_random_solution()

    while True:
        # get if the current solution is the optima, as well as any improved
        is_optima, solution = local_optima(solution)
        best_distance = evaluate(solution)
        # solution found
        if is_optima:  # if the local optima has been found, exit the loop
            break
        else:
            # create a new random neighbouring solution to test
            neighbour = random_neighbour_ryan(solution)

            # evaluate the newly created neighbour
            neighbour_distance = evaluate(neighbour)

            # if the neighbouring solution has a better distance than the current best distance
            if neighbour_distance < best_distance:
                solution = neighbour
                best_distance = neighbour_distance
                # print("new neighbour: ", best_distance)

    return solution, evaluate(solution)


def random_hill_climbing(num):
    random_sol = generate_random_solution()

    print("Initial Permutation:", random_sol)

    best_distance = evaluate(random_sol)
    print("Initial Distance:", best_distance)

    solution = random_sol.copy()

    iteration = 0

    while iteration < num:

        neighbour = random_neighbour_ryan(solution)

        neighbour_distance = evaluate(neighbour)

        if neighbour_distance < best_distance:
            solution = neighbour.copy()

            best_distance = neighbour_distance

            # print("New Permutation:", sol[1])
            # print("New Distance:", evaluate(sol))
            # print("New Sol:", sol)

        iteration += 1
    print("Final distance: ", best_distance)
    # plot_colours(test_colours, solution)
    return solution, evaluate(solution)


# method to perform multiple hill climbs, taking in the number of tries to be attempted
def multi_hill_climb(tries):
    distances = []
    best_sol = []
    best_distance = random_hill_climbing_local_optima()[1]  # random high value, since we are trying to find a minimum

    for i in range(0, tries - 1):
        hill_climb = random_hill_climbing_local_optima()
        distances.append(hill_climb[1])
        if hill_climb[1] < best_distance:
            best_distance = hill_climb[1]
            best_sol = hill_climb[0]

    print("from", tries, "Multi-HC found", "\nthe best distance of: ", best_distance,
          "\nusing solution: ", best_sol)
    plot_colours(test_colours, best_sol)


def multi_hill_climb_ryan(iter):
    solutions = []

    for i in range(iter):
        solutions.append(random_hill_climbing(2000)[0])

    best_sol = solutions[0]
    print(best_sol)
    k = evaluate(solutions[0])

    for j in range(len(solutions)):
        print(solutions[j])
        if evaluate(solutions[j]) < k:
            k = evaluate(solutions[j])
            best_sol = copy.deepcopy(solutions[j])

    print(best_sol)

    print("Best Sol:", evaluate(best_sol))

    plot_colours(test_colours, best_sol)

    return best_sol


def organise_avg(start_solution):
    sorted_solution = start_solution.copy()

    avg_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        avg_col[j] = sum(test_colours[sorted_solution[j]]) / len(test_colours[sorted_solution[j]])

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if avg_col[l] > avg_col[l + 1]:
                temp = avg_col[l]

                avg_col[l] = avg_col[l + 1]
                avg_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def organise_red(start_solution):
    sorted_solution = start_solution.copy()

    red_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        red_col[j] = test_colours[sorted_solution[j]][0]

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if red_col[l] < red_col[l + 1]:
                temp = red_col[l]

                red_col[l] = red_col[l + 1]
                red_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def organise_green(start_solution):
    sorted_solution = start_solution.copy()

    green_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        green_col[j] = test_colours[sorted_solution[j]][1]

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if green_col[l] < green_col[l + 1]:
                temp = green_col[l]

                green_col[l] = green_col[l + 1]
                green_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def organise_blue(start_solution):
    sorted_solution = start_solution.copy()

    blue_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        blue_col[j] = test_colours[sorted_solution[j]][2]

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if blue_col[l] < blue_col[l + 1]:
                temp = blue_col[l]

                blue_col[l] = blue_col[l + 1]
                blue_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def organise_ratio_red(start_solution):
    sorted_solution = start_solution.copy()

    ratio_red_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        ratio_red_col[j] = test_colours[sorted_solution[j]][0] / (test_colours[sorted_solution[j]][1]
                                                            + test_colours[sorted_solution[j]][2])

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if ratio_red_col[l] < ratio_red_col[l + 1]:
                temp = ratio_red_col[l]

                ratio_red_col[l] = ratio_red_col[l + 1]
                ratio_red_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def organise_ratio_green(start_solution):
    sorted_solution = start_solution.copy()

    ratio_red_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        ratio_red_col[j] = test_colours[sorted_solution[j]][1] / (test_colours[sorted_solution[j]][0]
                                                                  + test_colours[sorted_solution[j]][2])

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if ratio_red_col[l] < ratio_red_col[l + 1]:
                temp = ratio_red_col[l]

                ratio_red_col[l] = ratio_red_col[l + 1]
                ratio_red_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm
    return sorted_solution


def organise_ratio_blue(start_solution):
    sorted_solution = start_solution.copy()

    ratio_red_col = sorted_solution.copy()

    for j in range(len(sorted_solution)):
        ratio_red_col[j] = test_colours[sorted_solution[j]][2] / (test_colours[sorted_solution[j]][1]
                                                                  + test_colours[sorted_solution[j]][0])

    for k in range(len(sorted_solution)):

        for l in range(len(sorted_solution) - k - 1):

            if ratio_red_col[l] < ratio_red_col[l + 1]:
                temp = ratio_red_col[l]

                ratio_red_col[l] = ratio_red_col[l + 1]
                ratio_red_col[l + 1] = temp

                temp_perm = sorted_solution[l]

                sorted_solution[l] = sorted_solution[l + 1]
                sorted_solution[l + 1] = temp_perm

    return sorted_solution


def loc_opt():

    original = generate_random_solution()

    best_distance = evaluate(original)

    temp_pos1 = 0
    temp_pos2 = 0
    better_found = True

    sol = original.copy()

    while better_found:

        better_found = False

        for i in range(len(sol) - 1):

            temp1 = sol[i]
            temp2 = sol[i+1]

            sol[i] = temp2
            sol[i+1] = temp1

            new_distance = evaluate(sol)

            if new_distance < best_distance:
                better_found = True
                best_distance = new_distance
                temp_pos1 = i
                temp_pos2 = i + 1

            sol = original.copy()

        temp1 = sol[temp_pos1]
        temp2 = sol[temp_pos2]

        sol[temp_pos1] = temp2
        sol[temp_pos2] = temp1

    return sol


#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 100  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

# permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
permutation = random.sample(range(test_size),
                            test_size)  # produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

test = generate_random_solution()
# random_hill_climbing(1000)
# multi_hill_climb_ryan(20)
# multi_hill_climb(20)
solve = solve(test)
average = organise_avg(test)
red = organise_red(test)
green = organise_green(test)
blue = organise_blue(test)
ratio_red = organise_ratio_red(test)
ratio_green = organise_ratio_green(test)
ratio_blue = organise_ratio_blue(test)

plot_colours(test_colours, solve)
print("solve achieved: ", evaluate(solve))
plot_colours(test_colours, average)
print("average achieved: ", evaluate(average))
plot_colours(test_colours, red)
print("red achieved: ", evaluate(red))
plot_colours(test_colours, green)
print("green achieved: ", evaluate(green))
plot_colours(test_colours, blue)
print("blue achieved: ", evaluate(blue))
plot_colours(test_colours, ratio_red)
print("ratio red: ", evaluate(ratio_red))
plot_colours(test_colours, ratio_green)
print("ratio green: ", evaluate(ratio_green))
plot_colours(test_colours, ratio_blue)
print("ratio blue: ", evaluate(ratio_blue))

# multi_hill_climb(100)
# multi_hill_climb(1000)
# multi_hill_climb(5000)
# multi_hill_climb(25000)

opt = loc_opt()

plot_colours(test_colours, opt)



print("loc opt achieved", evaluate(opt))

temp = local_optima(opt)

print(temp[0], "Better solution found with distance:", evaluate(temp[1]))
exit()
