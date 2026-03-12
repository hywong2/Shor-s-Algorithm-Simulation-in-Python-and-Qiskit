# Author: Trung Nguyen
# Affiliation: MSQT Student, San Jose State University
# Contact: trung.nguyen03@sjsu.edu
# Date: February 2026

"""
The code implements Shor algorithm by direct matrix multiplication.
"""
import numpy as np
from fractions import Fraction
from math import gcd
from  shor_helper import *

# Gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)    # Hadamard gate
Id = np.eye(2)                                  # Identity gate

# Full Permutation Matrix Operator Uai |y> = y.a^{2^i} (mod N)
# Handle special case of reduced qubit n_target < N
def permute_power_mod_N(a, i, N, n_target=4):
    power = 2**i
    b = pow(a, power, N)  # a^{2^i} mod N
    size = 2**n_target       # 16 for N=15
    U_ai = np.zeros((size, size), dtype=complex)
    for y in range(size):
        if y < N:  # Only affect states < N
            new_y = (y * b) % N
            if new_y >= size:	# handle case for reduced n_t
                new_y = y		# for new_y > 2**n, Inserting Identity 
        else:
            new_y = y  # Insterting identity
        U_ai[new_y, y] = 1.0
    return U_ai

# generate IQFT matrix operator  of dimension n
def iqft(n):
    IQFT = np.zeros((2**n, 2**n), dtype=complex)
    for x in range(2**n):
        for y in range(2**n):
            IQFT[x, y] = np.exp(-1j * 2*np.pi * x * y / 2**n) / np.sqrt(2**n)
    return IQFT

def measure_target(state, n_control, n_target):
    """ Returns:
        measured_y        : the collapsed target value 
        post_control_state: normalized state vector after measurement
        prob_y            : probability of getting y-value
    """
    dim_target = 1 << n_target		# store statevector of target reg
    dim_control  = 1 << n_control	# store statevector of control reg
    
    # Compute marginal probs for each |y> of target register
    marginal_y = np.zeros(dim_target)
    for y in range(dim_target):
        indices = np.arange(y, len(state), dim_target) 		
        marginal_y[y] = np.sum(np.abs(state[indices])**2)	
       
    # Measure target register
    measured_y = np.random.choice(dim_target, p=marginal_y / np.sum(marginal_y))
    
    # Update the statevector
    indices_y = np.arange(measured_y, len(state), dim_target) 
    temp_state = np.zeros_like(state)
    temp_state[indices_y] = state[indices_y]
    state[:] = temp_state[:]				

    # normalize statevector
    norm = np.linalg.norm(state)			
    if norm > 1e-12:
        state /= norm
    
    return measured_y, marginal_y[measured_y]
    
# Shor algorithm realization
def shor_algorithm(N, a, n_target, n_control, plot):
    """ Returns:  measured_x (the collapsed control value (integer))
        Display states and probs
        Plot marginal probability of x, y (On/Off).
    """

    # Step 1a: Initialize total statevector (MSB |reg1> = |0>, |reg2> = |1> LSB)
    total_size = 2**(n_control + n_target)                
    state = np.zeros(total_size, dtype=complex) # state vector of 2**total_size bases
    state[1] = 1.0                              # |1> in target register (LSB)

    print("\nStep 1a: State vector at initialization")
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)

    # Step 1b: Hadamard on Control qubits
    H_count = np.eye(1)
    for i in range(n_control):
        H_count = np.kron(H, H_count   )                  # H  x n_control time
    H_full = np.kron(H_count, np.eye(2**n_target))  
    state = H_full @ state

    print("\nStep 1b: State vector after Hadarmard")
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label= f"Step 1: State vector after initialization (N={N}, a={a})")    

    # Projectors
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    
    # Step 2: Apply Oracle Uf by multiplication of controlled U 
    # (from i=0: lowest power 2^0, lowest qubit of control_qubits)
    for i in range(n_control):
        U_mult = permute_power_mod_N(a, i, N, n_target)
        I_above = np.eye(2**(n_control - 1 - i))
        I_below_count = np.eye(2**i)
        I_n_target = np.eye(2**n_target)
        controlled_block = np.kron(P0, np.kron(I_below_count, I_n_target)) + \
                                      np.kron(P1, np.kron(I_below_count, U_mult))
        global_controlled_U = np.kron(I_above, controlled_block)
        
        state = global_controlled_U @ state

    print("\nStep 2: State vector after applying Oracle Uf")
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label=f"Step 2: State vector after applying Oracle Uf (N={N}, a={a})")    
    
    # Step 3: Inverse QFT on Control
    iqft_mat = iqft(n_control)
    iqft_full = np.kron(iqft_mat, np.eye(2**n_target))
    state = iqft_full @ state
    
    print("\nStep 3: State vector after applying iQFT")
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label=f"Step 3: State vector after applying iQFT (N={N}, a={a})")    

    # Step 4: Measure target register 
    y_measured, y_prob = measure_target(state, n_control, n_target)
    print("\nStep 4: State vector after measuring target register")
    print(f"Measured target register: |{format(y_measured, f'0{n_target}b')}>  (probability was {y_prob:.4f})")
    print_state_vector(state, n_control, n_target, threshold=0.001, top_k=20)
    if plot:
        plot_marginal_quantum_state(state, n_control, n_target, label= f"Step 4: State vector after measuring target register (N={N}, a={a}")    

    # Step 5: Measure control register
    measured_state = np.random.choice(1 << (n_control+n_target), p=np.abs(state)**2)
    measured_x = measured_state >> n_target
    x_prob = abs(state[measured_state])**2
    print("\nStep 5: State vector after measuring control register")
    print(f"Measured control register: |{format(y_measured, f'0{n_control}b')}>  (probability was {x_prob:.4f})")

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
        print(f"Result: R = {r}")
        
        if measured_x != 0 and r % 2 == 0:
            guesses = [gcd(pow(a, r//2) - 1, N), gcd(pow(a, r//2) + 1, N)]
            print(f"Guessed Factors: {guesses[0]} and {guesses[1]}")
            for guess in guesses:
                if 1 < guess < N:
                    print(f"*** Non-trivial factor found: {guess} ***")
                    factor_found = True

# Main program - should run only with N=15, can modify a, n_target, n_control

#shor_factorization(15, 7, 4, 3, True)

#shor_factorization(15, 2, 4, 3, True)

shor_factorization(15, 7, 4, 8, True)
