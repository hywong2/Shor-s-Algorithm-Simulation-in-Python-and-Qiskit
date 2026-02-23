import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd
from shor_helper import *

def get_permute_indices(a, i, N, n_target):
    """
    U|y> = |(y * a^(2^i)) mod N>
    Returns a 1D mapping array for the target register.
    """
    power = pow(a, 2**i, N)               # compute a^(2^i)) mod N
    indices = np.arange(2**n_target)      # identity mapping table, instead of matrix
    valid_mask = indices < N              # create mask for permutation
    indices[valid_mask] = (indices[valid_mask] * power) % N
    return indices

def shor_algorithm(N, a, n_target, n_control, plot):

    n_c_size, n_t_size = 2**n_control, 2**n_target

    # Step 1: Initialize
    # state shape (Control, Target)
    state_2d = np.zeros((n_c_size, n_t_size), dtype=complex)
    state_2d[:, 1] = 1.0 / np.sqrt(n_c_size) # Apply H on control, |1> on target

    print("\nStep 1a: State vector at initialization")
    state = state_2d.flatten()
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label=f"Step 1: State vector after initialization (N={N}, a={a})")

    # Step 2: Oracle Uf (The Slicing/View Optimization)
    for i in range(n_control):
        perm_map = get_permute_indices(a, i, N, n_target)
        # Reshape to isolate the i-th control qubit
        view_shape = (2**(n_control - 1 - i), 2, 2**i, n_t_size)
        state_view = state_2d.reshape(view_shape)
        
        # Apply permutation only where the control qubit is |1>
        state_view[:, 1, :, :] = state_view[:, 1, :, :][..., perm_map]

    print("\nStep 2: State vector after applying Oracle Uf")
    state = state_2d.flatten()
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label=f"Step 2: State vector after applying Oracle Uf (N={N}, a={a})")    

    # Step 3: Inverse QFT (FFT Optimization)
    # Applied along the control register axis
    state_2d = np.fft.ifft(state_2d, axis=0) * np.sqrt(n_c_size)

    print("\nStep 3: State vector after applying iQFT")
    state = state_2d.flatten()
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label=f"Step 3: State vector after applying iQFT (N={N}, a={a})")    

    # Step 4-5: Measure control register directly
    prob_c = np.sum(np.abs(state_2d)**2, axis=1)
    measured_x = np.random.choice(n_c_size, p=prob_c/np.sum(prob_c))
    
    return measured_x

# Shor post-processing
def shor_factorization(N, a, n_target, n_control, plot = False):
    factor_found = False
    attempt = 0
    while not factor_found:
        attempt += 1
        print(f"\nAttempt {attempt}:")
        measured_x = shor_algorithm(N, a, n_target, n_control, plot)
        frac = Fraction(measured_x/2**n_control).limit_denominator(N)   # continued fraction algorithm
        r = frac.denominator
        print(f"Result: r = {r}")
        
        if measured_x != 0 and r % 2 == 0:
            guesses = [gcd(pow(a, r//2) - 1, N), gcd(pow(a, r//2) + 1, N)]
            print(f"Guessed Factors: {guesses[0]} and {guesses[1]}")
            for guess in guesses:
                if 1 < guess < N:
                    print(f"*** Non-trivial factor found: {guess} ***")
                    factor_found = True

# Main program - modify N, a, n_target and n_control
shor_factorization(N=21, a=11, n_target = 5, n_control=10, plot=True)
