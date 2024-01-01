# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:42:18 2024

@author: 52331
"""

import numpy as np
import matplotlib.pyplot as plt

def markov_simulation_one_individual(transition_matrix, initial_vector, T):
    # Validate inputs
    n = len(transition_matrix)
    if n != len(initial_vector) or not all(len(row) == n for row in transition_matrix):
        raise ValueError("Invalid input dimensions")

    # Perform simulation
    current_state = np.copy(initial_vector)
    historical_states = np.zeros(T)

    choices = np.arange(n)
    for t in range(T):
        # Choose next state based on transition probabilities
        
        next_state = np.random.choice(choices, p=current_state)
        print("Iteration {}: {}".format(t, next_state))
        
        historical_states[t] = next_state
        
        # Update current state for the next iteration
        current_state = transition_matrix[next_state]
    
    return historical_states

# Example using transition matrix from Maehlmann Thomas, 2006
# Define the transition matrix and initialization vector
transition_matrix = np.array([
    [0.6614, 0.2547, 0.0417, 0.0309, 0.0058, 0.0035, 0.0020],
    [0.0638, 0.6802, 0.1798, 0.0599, 0.0124, 0.0034, 0.0005],
    [0.0139, 0.2363, 0.4737, 0.2246, 0.0397, 0.0106, 0.0012],
    [0.0068, 0.0784, 0.2341, 0.4960, 0.1466, 0.0344, 0.0037],
    [0.0025, 0.0343, 0.0809, 0.2870, 0.4432, 0.1332, 0.0189],
    [0.0007, 0.0147, 0.0460, 0.0896, 0.2406, 0.4625, 0.1459],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
])

initial_vector = np.array([0.06984022, 0.30826882, 0.24098171, 0.22969857, 0.11077253,
       0.04043815, 0]) # Initial classification of any company is not in default and respects the distribution of the rest of the categories in the paper

# Set the number of iterations
T = 300

# Run the simulation
for _ in range(20):
    hstates = markov_simulation_one_individual(transition_matrix, initial_vector, T)
    plt.plot(hstates)
