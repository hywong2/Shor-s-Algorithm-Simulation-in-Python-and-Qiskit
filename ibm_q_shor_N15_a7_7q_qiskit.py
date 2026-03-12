# Author: Trung Nguyen
# Affiliation: MSQT Student, San Jose State University
# Contact: trung.nguyen03@sjsu.edu
# Date: February 2026

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Connection ---
# Using standard 'ibm_quantum' channel and your provided token
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="_jupWqgkeDvgXWqTlUryoFJsTQrNb6UB-n30tHKrZUxe")
backend = service.backend("ibm_fez")

# --- 2. Circuit Building ---
y = QuantumRegister(4, 'y')
x = QuantumRegister(3, 'x')
c = ClassicalRegister(7, 'c')
qc = QuantumCircuit(y, x, c)

# Necessary so that modular multiplication (7^x * y) has a base to work from.
qc.x(y[0])

# This allows the circuit to calculate all powers of 7 mod 15 simultaneously.
qc.h(x)
qc.barrier()

# Stage 1: Multiplication by 7 (Controlled by x[0])
qc.cx(x[0], y[0]); qc.cx(x[0], y[1]); qc.cx(x[0], y[2]); qc.cx(x[0], y[3])
qc.cswap(x[0], y[0], y[3]); qc.cswap(x[0], y[0], y[1]); qc.cswap(x[0], y[2], y[1])
qc.barrier()

# Stage 2: Multiplication by 49 mod 15 = 4 (Controlled by x[1])
qc.cswap(x[1], y[2], y[0]); qc.cswap(x[1], y[1], y[3])
qc.barrier()

# --- INVERSE QFT on the Control Register (x) ---
qc.h(x[2])
qc.cp(-np.pi/2, x[1], x[2])
qc.cp(-np.pi/4, x[0], x[2])
qc.h(x[1])
qc.cp(-np.pi/2, x[0], x[1])
qc.h(x[0])
qc.swap(x[0], x[2])

# Measurements: Mapping x register to bits 4, 5, and 6
qc.measure(x, c[4:7]) 

# --- 3. Transpile & Run ---
# Optimization level 3 is crucial to compress the heavy CSWAP logic for hardware
tqc = transpile(qc, backend=backend, optimization_level=3)
sampler = Sampler(mode=backend)

print(f"Submitting job to {backend.name}...")
job = sampler.run([tqc], shots=1024)

print(f"Job ID: {job.job_id()}")
print(f"Current Status: {job.status()}")

# --- 4. Result Processing ---
result = job.result()
counts = result[0].data.c.get_counts()

# Process only the x-register bits
x_counts = {}
for bitstring, count in counts.items():
    # Slicing the 3 leftmost bits (representing the x register)
    x_val = bitstring[0:3] 
    x_counts[x_val] = x_counts.get(x_val, 0) + count

# Final Visualization
print("Final X-Register Counts:", x_counts)
plot_histogram(x_counts, title=f"Shor's N=15, a=7 (x-register) on {backend.name}")
plt.show()

# Draw Logical Circuit
qc.draw("mpl")
plt.show()

tqc.draw("mpl", cregbundle=True, idle_wires=False)
plt.show()
