import math

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
import statistics as st
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

    results = []

    random_sol = solve(generate_random_solution())

    print("Initial Permutation:", random_sol)

    best_distance = evaluate(random_sol)
    print("Initial Distance:", best_distance)

    results.append(best_distance)

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

        results.append(best_distance)
    print("Final distance: ", best_distance)

    plt.title('Hill Climbing Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    plt.plot(results)
    plt.show()

    plot_colours(test_colours, solution)

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
    results = []

    for i in range(iter):
        solutions.append(random_hill_climbing(2000)[0])

    best_sol = solutions[0]
    print(best_sol)
    k = evaluate(solutions[0])

    for j in range(len(solutions)):
        # To calculate mean, median and sd
        results.append(evaluate(solutions[j]))
        if evaluate(solutions[j]) < k:
            k = evaluate(solutions[j])
            best_sol = copy.deepcopy(solutions[j])

    mean = sum(results)/len(results)
    median = st.median(results)
    standard_dev = st.pstdev(results)

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation", standard_dev)

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


# Calculates local optima by taking the best neighbour it can find by swapping adjacent indices
def loc_opt(sol):

    best_distance = evaluate(sol)

    temp_pos1 = 0
    temp_pos2 = 0
    better_found = True

    while better_found:

        better_found = False

        for i in range(len(sol) - 1):

            temp1 = sol[i]
            temp2 = sol[i + 1]

            sol[i] = temp2
            sol[i + 1] = temp1

            new_distance = evaluate(sol)

            if new_distance < best_distance:
                best_distance = new_distance
                temp_pos1 = i
                temp_pos2 = i + 1
                better_found = True

            sol[i] = temp1
            sol[i + 1] = temp2

        if better_found:
            temp1 = sol[temp_pos1]
            temp2 = sol[temp_pos2]

            sol[temp_pos1] = temp2
            sol[temp_pos2] = temp1

    return sol


# Calculates local optima by taking the best neighbour it can find by searching all possible neighbours
def loc_opt2():

    sol = generate_random_solution()

    best_distance = evaluate(sol)

    temp_pos1 = 0
    temp_pos2 = 0
    better_found = True

    while better_found:

        better_found = False

        for i in range(len(sol) - 1):

            for j in range(i + 1, len(sol)):

                temp1 = sol[i]
                temp2 = sol[j]

                sol[i] = temp2
                sol[j] = temp1

                new_distance = evaluate(sol)

                if new_distance < best_distance:
                    best_distance = new_distance
                    temp_pos1 = i
                    temp_pos2 = j
                    better_found = True

                sol[i] = temp1
                sol[j] = temp2

        if better_found:
            temp1 = sol[temp_pos1]
            temp2 = sol[temp_pos2]

            sol[temp_pos1] = temp2
            sol[temp_pos2] = temp1

    return sol


# Calculates local optima by taking the first better neighbour it finds
def loc_opt3():

    sol = solve(generate_random_solution())

    best_distance = evaluate(sol)

    print(best_distance)

    temp_pos1 = 0
    temp_pos2 = 0
    better_found = True

    while better_found:

        better_found = False

        for i in range(len(sol) - 1):

            for j in range(i + 1, len(sol)):

                temp1 = sol[i]
                temp2 = sol[j]

                sol[i] = temp2
                sol[j] = temp1

                new_distance = evaluate(sol)

                if new_distance < best_distance:
                    best_distance = new_distance
                    temp_pos1 = i
                    temp_pos2 = j
                    better_found = True

                sol[i] = temp1
                sol[j] = temp2

                if better_found:
                    break

        if better_found:
            temp1 = sol[temp_pos1]
            temp2 = sol[temp_pos2]

            sol[temp_pos1] = temp2
            sol[temp_pos2] = temp1

            print(best_distance)

    return sol


def hsl():
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


        # Calculating Lum and Sat
        '''lum = (max_value + min_value) / 2

        if max_value == min_value:
            hue = 0
        else:
            if lum >= 0.5:
                sat = (max_value - min_value) / (max_value + min_value)
            else:
                sat = (max_value - min_value) / (2 - max_value - min_value)'''

        # One source said mod 6 and another didn't, mod 6 returns a better distance
        if max_index == 0:
            hue = ((green - blue))/(max_value - min_value)

        if max_index == 1:
            hue = 2 + (blue - red)/(max_value - min_value)

        if max_index == 2:
            hue = 4 + (red - green)/(max_value - min_value)

        hue = hue * 60

        if max_value == min_value:
            hue = 0

        if hue < 0:
            hue += 360

        hsl.append(hue)

    for k in range(len(solution)):

        for l in range(len(solution) - k - 1):

            if hsl[l] > hsl[l + 1]:
                temp = hsl[l]

                hsl[l] = hsl[l + 1]
                hsl[l + 1] = temp

                temp_perm = solution[l]

                solution[l] = solution[l + 1]
                solution[l + 1] = temp_perm


    return solution


def iterator(num):

    #Choose starting solution
    best_sol = loc_opt(hsl())

    it = 0

    while it < num:

        temp_sol = random_neighbour(best_sol)

        temp_sol2 = loc_opt(temp_sol)

        if evaluate(temp_sol2) < evaluate(best_sol):
            best_sol = temp_sol2

        it +=1

    return best_sol


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


random_hill_climbing(1000000)
#multi_hill_climb_ryan(30)


'''test = generate_random_solution()
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

opt2 = loc_opt2()
plot_colours(test_colours, opt2)
print("loc opt2 achieved", evaluate(opt2))

opt3 = loc_opt3()
plot_colours(test_colours, opt3)
print("loc opt3 achieved", evaluate(opt3))

print(local_optima(opt3))'''


'''test = hsl()
plot_colours(test_colours, test)

print("hsl achieved:",evaluate(test))

opt = loc_opt3()
plot_colours(test_colours, opt)
print("loc opt achieved", evaluate(opt))'''

a = generate_random_solution()

ind = a.index(893)

print(a)
print(ind)

teste = test_colours[a[ind]]

print("teste",teste)

test = hsl()

plot_colours(test_colours, test)

#WTF?//TODO
print(test_colours[test[17]])
print(test_colours[test[30]])
print(test_colours[test[30]])

#THIS IS 30?!?!
print(test[39])
print(test)

print(test[0])


exit()
