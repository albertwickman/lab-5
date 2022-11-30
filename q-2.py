import numpy as np
from copy import deepcopy
from random import randrange

# Actions, directions are N, S, E and W for North, South, East and West
A = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0)
}


# Class for creating a cell
class Cell:
    def __init__(self, value, reward, position):
        self.value = value
        self.reward = reward
        self.position = position
        self.policy = position
        self.A = []
        self.set_A()

    def set_A(self):  # Sets all the directions
        if self.position[1] > 0:
            self.A.append('S')
        if self.position[1] < 2:
            self.A.append('N')
        if self.position[0] > 0:
            self.A.append('W')
        if self.position[0] < 2:
            self.A.append('E')


# Makes a matrix with values
matrix = [[Cell(randrange(10), 0, (i, 2 - k))
           for i in range(3)]
          for k in range(3)]

matrix_middle = matrix[1][1]

# Reward = 10 for indices x = 1 and y = 1
matrix_middle.reward = 10

# Epsilon, gamma and the probability
eps = 0.1
gam = 0.9
prob = 0.8

# Number of iterations. Gets incremented in the while loop
n_iter = 0


# Prints the matrix
def print_matrix(matrix):
    print('Values of the convergence matrix')

    # range(3) since it is a 3x3 matrix, rounds the numbers to two decimals
    for r in range(3):
        print(round(matrix[r][0].value, 2), round(matrix[r][1].value, 2), round(matrix[r][2].value, 2))

    print('\nThe optimal policy of the matrix')
    for r in range(3):
        print(matrix[r][0].policy, matrix[r][1].policy, matrix[r][2].policy)


# Gets a cell by x and y coordinates
def get_cell(matrix, x_coord, y_coord):
    return matrix[2 - x_coord][y_coord]


while True:
    highest_prob = 0

    # Deepcopy instead of regular copying to steer clear of problems regarding references
    updated_matrix = deepcopy(matrix)

    for r in matrix:
        for state in r:
            x_coord, y_coord = state.position
            prev_value = state.value
            updated_value = 0

            # Iterative attempts of different actions
            for action in state.A:
                # Updates the coordinates after a step/action has been made
                updated_x_coord = state.position[0] + A[action][0]
                updated_y_coord = state.position[1] + A[action][1]

                next_state = get_cell(matrix, updated_x_coord, updated_y_coord)

                value = prob * (next_state.reward + gam * next_state.value) + (1 - prob) * (state.reward + gam * state.value)

                # Store the value if it gives a higher value, used for improving the policy
                if value > updated_value:
                    updated_value = value
                    updated_matrix[x_coord][y_coord].policy = next_state.position

            # Updates the matrix to the new values and gives the highest probability
            updated_matrix[x_coord][y_coord].value = updated_value
            highest_prob = max(highest_prob, np.abs(prev_value - updated_value))

    if highest_prob < eps:
        break
    n_iter = n_iter + 1

    matrix = deepcopy(updated_matrix)

print_matrix(matrix)
print('\nNumber of iterations:', n_iter)
