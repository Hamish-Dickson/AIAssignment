import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
import statistics as st
from math import sqrt


# TODO Write an introduction? Introduce that a solution is a permutation of the colours in the file?

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

    ratio = 50  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(col), 3))
    for i in range(0, len(col)):
        img[:, i, :] = colours[perm[i]]

    fig, axes = plt.subplots(1, figsize=(20, 10))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest', aspect='auto')
    axes.axis('off')
    plt.savefig("plot.png")
    plt.show()


# Calculates distance between 2 colours using euclidean distance formula
def calculate_distance(colour1, colour2):
    return sqrt(((colour2[0] - colour1[0]) ** 2) + ((colour2[1] - colour1[1]) ** 2)
                + ((colour2[2] - colour1[2]) ** 2))


# Returns the total distance between colours in a given solution
def evaluate(solution):
    total_distance = 0

    for i in range(0, len(solution) - 1):
        # print(colour_one, colour_two)
        total_distance += calculate_distance(test_colours[solution[i]], test_colours[solution[i + 1]])

    return total_distance


# Generates a random solution (permutation of colours)
def generate_random_solution():
    # permutation is simply order of elements to be chosen I.E 0, 1, 4, 5. could change to 0, 1, 2, 3 for testing
    random_solution = random.sample(range(test_size),
                                    test_size)  # produces random pemutation of length test_size, from the numbers 0 to test_size -1

    return random_solution


# Iterates through each colour and finds the proceeding colour with the shortest distance between them and swaps it with
# the colour next to itself
def nearest_neighbour(start_solution):
    sorted_solution = start_solution.copy()

    # The index where the current closest colour is stored
    temp_space = 0

    # Has to have the -1 as we are checking j+1
    for j in range(len(sorted_solution) - 1):

        smaller_found = False
        # Finds the distance between the starting colour and the colour next to it to use as the starting point
        smallest_dist = calculate_distance(test_colours[sorted_solution[j]], test_colours[sorted_solution[j + 1]])

        # Uses j+2 as at the beginning of this iteration j+1 was already checked for the starting distance
        for k in range(j + 2, len(sorted_solution)):

            dist = calculate_distance(test_colours[sorted_solution[j]], test_colours[sorted_solution[k]])

            # If the distance between the starting colour and the most recently checked colour is smaller than the
            # previous smallest distance, it becomes the new smallest distance and the indices for the two colours are
            # stored.
            if dist < smallest_dist:
                smallest_dist = dist
                smaller_found = True

                temp_space = k
                temp_pos2 = sorted_solution[k]

        # Once the current beginning colour has gone through all colours after it, if a smaller distance was found than
        # the one it started with, the indices are swapped
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

    # Chooses two random numbers to get the selection to reverse
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


# Calculates a random neighbour of random solution, if the neighbour has a smaller total distance, it becomes the
# solution, this is repeated for the specified number of iterations

def random_hill_climbing(num):
    results = []
    random_sol = generate_random_solution()

    # Starting distance
    best_distance = evaluate(random_sol)
    results.append(best_distance)

    solution = random_sol.copy()
    iteration = 0

    while iteration < num:

        neighbour = random_neighbour_ryan(solution)

        neighbour_distance = evaluate(neighbour)

        if neighbour_distance < best_distance:
            solution = neighbour.copy()

            best_distance = neighbour_distance

        iteration += 1

        results.append(best_distance)

    '''plt.title('Hill Climbing Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.plot(results)
    plt.show()'''

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


# Runs the specified number of hillclimbs, each hillclimb returns it's best solution which is then added to a list, the
# best result of all of the hillclimbs is returned.
# Additionally, the list of best solutions is used to generate the mean, median and standard deviation of the results.
def multi_hill_climb_ryan(number_of_hillclimbs, hillclimb_type):
    solutions = []
    results = []

    if hillclimb_type.lower() == 'random':
        for i in range(number_of_hillclimbs):
            solutions.append(random_hill_climbing(2000)[0])
    elif hillclimb_type.lower() == 'knn':
        for i in range(number_of_hillclimbs):
            solutions.append(nearest_neighbour(generate_random_solution()))
    else:
        print('Incorrect usage for multi hill climb.')
        exit(1)


    best_sol = solutions[0]
    print(best_sol)
    k = evaluate(solutions[0])

    for j in range(len(solutions)):
        # To calculate mean, median and sd
        results.append(evaluate(solutions[j]))
        if evaluate(solutions[j]) < k:
            k = evaluate(solutions[j])
            best_sol = copy.deepcopy(solutions[j])

    mean = sum(results) / len(results)
    median = st.median(results)
    standard_dev = st.pstdev(results)

    print("Results for " + hillclimb_type + " algorithm")
    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation", standard_dev)

    print("Best permutation: ", best_sol)

    print("Best Sol:", evaluate(best_sol))

    plt.title(hillclimb_type + ' Hill climb for ' + str(test_size) + ' colours. ' + str(number_of_hillclimbs) + ' iterations.')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.plot(results)
    plt.show()

    plot_colours(test_colours, best_sol)

    return best_sol, results


# Organises the colours based on the average of their red, green and blue values
def organise_avg():
    sorted_solution = generate_random_solution()

    avg_col = []

    # Appends to avg_col the average of the red, green and blue values
    for j in range(len(sorted_solution)):
        avg_col.append(sum(test_colours[sorted_solution[j]]) / len(test_colours[sorted_solution[j]]))

    # Sorts the list of averages and the solution
    sort(sorted_solution, avg_col)

    return sorted_solution


# TODO make the following methods switches?

# Organises the colours based on their red value
def organise_red():
    sorted_solution = generate_random_solution()

    red_col = []

    # Appends to red_col the red value of the colour
    for j in range(len(sorted_solution)):
        red_col.append(test_colours[sorted_solution[j]][0])

    # Sorts the list of red values and the solution
    sort(sorted_solution, red_col)

    return sorted_solution


# Organises the colours based on their green value
def organise_green():
    sorted_solution = generate_random_solution()

    green_col = []

    # Appends to green_col the green value of the colour
    for j in range(len(sorted_solution)):
        green_col.append(test_colours[sorted_solution[j]][1])

    # Sorts the list of green values and the solution
    sort(sorted_solution, green_col)

    return sorted_solution


# Organises the colours based on their blue value
def organise_blue():
    sorted_solution = generate_random_solution()

    blue_col = []

    # Appends to blue_col the green value of the colour
    for j in range(len(sorted_solution)):
        blue_col.append(test_colours[sorted_solution[j]][2])

    # Sorts the list of blue values and the solution
    sort(sorted_solution, blue_col)

    return sorted_solution


# Organises the colours based on how much red is present in relation to green + blue
def organise_ratio_red():
    sorted_solution = generate_random_solution()

    ratio_red_col = []

    # Appends to ratio_red_col the value of the ratio of red to green + blue
    for j in range(len(sorted_solution)):
        ratio_red_col.append(test_colours[sorted_solution[j]][0] / (test_colours[sorted_solution[j]][1]
                                                                    + test_colours[sorted_solution[j]][2]))

    # Sorts the list of ratio values and the solution
    sort(sorted_solution, ratio_red_col)

    return sorted_solution


# Organises the colours based on how much green is present in relation to red + blue
def organise_ratio_green():
    sorted_solution = generate_random_solution()

    ratio_green_col = []

    # Appends to ratio_green_col the value of the ratio of green to red + blue
    for j in range(len(sorted_solution)):
        ratio_green_col.append(test_colours[sorted_solution[j]][1] / (test_colours[sorted_solution[j]][0]
                                                                      + test_colours[sorted_solution[j]][2]))

    # Sorts the list of ratio values and the solution
    sort(sorted_solution, ratio_green_col)
    return sorted_solution


# Organises the colours based on how much blue is present in relation to red + green
def organise_ratio_blue():
    sorted_solution = generate_random_solution()

    ratio_blue_col = []

    # Appends to ratio_blue_col the value of the ratio of blue to red + green
    for j in range(len(sorted_solution)):
        ratio_blue_col.append(test_colours[sorted_solution[j]][2] / (test_colours[sorted_solution[j]][1]
                                                                     + test_colours[sorted_solution[j]][0]))

    # Sorts the list of ratio values and the solution
    sort(sorted_solution, ratio_blue_col)

    return sorted_solution


# Converts the rgb values of the colours the colours hue and is then sorted based on the hue
def hue():
    hsl = []

    hue = 0

    solution = generate_random_solution()

    for i in range(len(solution)):

        red = test_colours[solution[i]][0]
        green = test_colours[solution[i]][1]
        blue = test_colours[solution[i]][2]

        min_value = min(test_colours[solution[i]])

        max_index = test_colours[solution[i]].index(max(test_colours[solution[i]]))
        max_value = max(test_colours[solution[i]])

        if max_index == 0:
            hue = (green - blue) / (max_value - min_value)

        if max_index == 1:
            hue = 2 + (blue - red) / (max_value - min_value)

        if max_index == 2:
            hue = 4 + (red - green) / (max_value - min_value)

        hue = hue * 60

        if max_value == min_value:
            hue = 0

        # As hue is measured between 0 and 360 degrees, if it is less than 0, 360 is added
        if hue < 0:
            hue += 360

        hsl.append(hue)

    # The list of hue values is sorted and so is the solution
    sort(solution, hsl)

    return solution


def plot_ryan(perm):
    colours = []
    cols = []

    for i in range(len(perm)):
        colours.append([test_colours[perm[i]][0], test_colours[perm[i]][1], test_colours[perm[i]][2]])

    cols.append(colours)

    fig, axes = plt.subplots(1, 1, figsize=(100, 20), dpi=10)

    axes.imshow(cols, interpolation='none', aspect='auto', origin='upper')
    axes.axis('off')
    # plt.savefig(name + '.png')
    plt.tight_layout()
    plt.show()


# A standard bubble sort
def sort(solution, colour_list):
    for k in range(len(solution)):

        for l in range(len(solution) - k - 1):

            if colour_list[l] > colour_list[l + 1]:
                temp = colour_list[l]

                colour_list[l] = colour_list[l + 1]
                colour_list[l + 1] = temp

                temp_perm = solution[l]

                solution[l] = solution[l + 1]
                solution[l + 1] = temp_perm


def demonstration():
    randomhc = multi_hill_climb_ryan(30, 'Random')
    knn = multi_hill_climb_ryan(30, 'KNN')

    plt.plot(randomhc[1], label='Random')
    plt.plot(knn[1], label='KNN')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.title('Side by side comparison of Random and KNN hillclimbs. ' + str(test_size) + ' colours.')
    plt.legend()
    plt.show()



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
'''
plot_ryan(organise_avg())
plot_ryan(organise_red())
plot_ryan(organise_green())
plot_ryan(organise_blue())
plot_ryan(organise_ratio_red())
plot_ryan(organise_ratio_green())
plot_ryan(organise_ratio_blue())
plot_ryan(hue())
plot_ryan(nearest_neighbour(generate_random_solution()))
'''
demonstration()

exit()
