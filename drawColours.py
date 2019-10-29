import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy


# Reading the colours
def read_file(fname):
    with open(fname, 'r') as afile:
        lines = afile.readlines()
    n = int(lines[3])  # number of colours  in the file
    col = []
    lines = lines[4:]  # colors as rgb values
    rgb = []
    for l in lines:
        rgb = l.split()

        for i in range(len(rgb)):
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
        img[:, i, :] = col[perm[i]]

    fig, axes = plt.subplots(1, figsize=(8, 4))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


# Calculates the Euclydian distance
def calculate_distance(colour1, colour2):
    d = math.sqrt(((colour2[0] - colour1[0]) ** 2) + ((colour2[1] - colour1[1]) ** 2)
                  + ((colour2[2] - colour1[2]) ** 2))
    return d


# Total up the distance between each colour and it's proceeding colour
def evaluate(solution):
    arranged_solution = arrange_colours(solution)

    total_distance = 0

    for i in range(0, len(solution[0]) - 1):
        total_distance += calculate_distance(arranged_solution[0][i], arranged_solution[0][i + 1])

    return total_distance


def arrange_colours(start_solution):
    random_cols = []

    # Creates and plots a random solution
    for i in range(len(start_solution[0])):
        random_cols.append(start_solution[0][start_solution[1][i]])

    arranged_sol = [random_cols, start_solution[1]]

    return arranged_sol


def generate_random_solution():
    # Get the directory where the file is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)  # Change the working directory so we can read the file4---------------------------------

    ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

    test_size = 1000  # Size of the subset of colours for testing
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


# Calculates a random neighbour by reversing a random section of the list.
def random_neighbour(solution):
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


def random_hill_climbing(num):
    random_sol = generate_random_solution()

    print("Initial Permutation:", random_sol[1])

    print("Initial:", random_sol)

    best_distance = evaluate(random_sol)
    print("Initial Distance:", best_distance)

    sol = solve(random_sol)

    k = 0

    while k < num:

        #CHANGE THIS
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

    for i in range(iter):
        sol.append(random_hill_climbing(4000))

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


y = multi_hill_climb(20)

plot_colours(y[0], y[1])

'''start = generate_random_solution()
print("initial distance:", evaluate(start))
plot_colours(start[0], start[1])

avg = organise_ratio_red(start)
print("avg distance:", evaluate(avg))
plot_colours(avg[0], avg[1])

best = solve(avg)
print("best distance:", evaluate(best))
plot_colours(best[0], best[1])

test = organise_red(best)
print("test distnace:", evaluate(test))
plot_colours(test[0], test[1])'''
