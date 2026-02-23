# Helper function
import matplotlib.pyplot as plt
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_marginal_quantum_state(state_vector, n_control, n_target, label="Quantum State"):
    """
    Plots marginal magnitudes and phases with independent x-axis scaling.
    
    Args:
        state_vector: Array-like, length 2**(n_control + n_target)
        n_control: Number of qubits at MSB
        n_target: Number of qubits at LSB
        label: String label to display on the plot title
    """
    # 1. Determine dimensions
    dim_control = 1 << n_control
    dim_target = 1 << n_target
    
    # 2. Reshape into (Rows=Control, Cols=Target) and get probabilities
    # We use abs()**2 to get the probability from complex amplitudes
    probs_2d = np.abs(state_vector).reshape((dim_control, dim_target))**2
    
    # 3. Calculate Marginals
    marginal_control = np.sum(probs_2d, axis=1)
    marginal_target = np.sum(probs_2d, axis=0)
    
    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(label, fontsize=16, fontweight='bold')
    
    # Define colors
    c_color = '#3498db'  # Nice Blue
    t_color = '#e74c3c'  # Nice Red
    
    # Plot Control Register Marginals
    # Setting edgecolor same as color removes the dark outline effect
    ax1.bar(range(dim_control), marginal_control, color=c_color, edgecolor=c_color)
    ax1.set_title(f"Control Register ({n_control} qubits)")
    ax1.set_xlabel("State Value (x)")
    ax1.set_ylabel("Marginal Probability")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Target Register Marginals
    ax2.bar(range(dim_target), marginal_target, color=t_color, edgecolor=t_color)
    ax2.set_title(f"Target Register ({n_target} qubits)")
    ax2.set_xlabel("State Value (y = f(x))")
    ax2.set_ylabel("Marginal Probability")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def print_state_vector(state, n_control, n_target, threshold=1e-8, top_k=None):
    """
    Prints the state vector in a readable format.
    
    Basis states: |x>|y> where
    - |x>: Control register (n_control qubits, higher bits)
    - |y>: target register (n_target qubits, lower bits)
    - Binary representation: MSB on the left
    
    Always displays both real and imaginary parts (even if imag =0).
    
    Parameters:
    - state: complex numpy array of length 2**(n_control + n_target)
    - n_control: number of Control qubits
    - n_target: number of target qubits
    - threshold: minimum probability to consider "significant" (when top_k=None)
    - top_k: if set, show exactly the top_k highest-probability states
    """
    total_size = len(state)
    expected_size = 2**(n_control + n_target)
    if total_size != expected_size:
        print(f"Error: State size {total_size} is different from expected {expected_size}")
        return
    
    probs = np.abs(state)**2
    sorted_indices = np.argsort(probs)[::-1]
    
    if top_k is None:
        display_indices = [idx for idx in sorted_indices if probs[idx] > threshold]
    else:
        display_indices = sorted_indices[:top_k]
    
    print(f"\nState Vector (|x> Control [{n_control} qubits] |y> target [{n_target} qubits], MSB left):")
    print(f"{'|x>':<{n_control+2}s} {'|y>':<{n_target+2}s} {'Amplitude (real + imag i)':<35s} Probability")
    print("-" * (n_control + n_target + 60))
    
    printed = 0
    for idx in display_indices:
        prob = probs[idx]
        if top_k is None and prob < threshold:
            continue
        
        amp = state[idx]
        real = amp.real
        imag = amp.imag
        
        # Always show both real and imag parts, with sign
        # Use + for positive, - for negative (including -0.0 → -0.0000000000)
        real_str = f"{real:+.10f}"
        imag_str = f"{imag:+.10f}"
        
        amp_str = f"{real_str} {imag_str}i"
        
        # Binary strings (MSB left)
        x_val = idx >> n_target
        y_val = idx & ((1 << n_target) - 1)
        x_bin = format(x_val, f'0{n_control}b')
        y_bin = format(y_val, f'0{n_target}b')
        
        print(f"|{x_bin}> |{y_bin}>   {amp_str:<35s} {prob:.10f}")
        printed += 1
    
    # Show remaining probability mass if any
    if top_k is not None:
        shown_prob = np.sum(probs[display_indices[:printed]])
        if 1.0 - shown_prob > 1e-8:
            print(f"... (remaining probability: {1.0 - shown_prob:.10f})")
    
    norm = np.linalg.norm(state)
    print(f"\nState vector norm: {norm:.12f} (should be ≈1.0)")
    print(f"Components shown: {printed}")

