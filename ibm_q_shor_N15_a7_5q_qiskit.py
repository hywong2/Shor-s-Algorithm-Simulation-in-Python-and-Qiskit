# Author: Trung Nguyen
# Affiliation: MSQT Student, San Jose State University
# Contact: trung.nguyen03@sjsu.edu
# Date: February 2026

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 1. Authenticate with IBM Quantum
MY_TOKEN = "_jupWqgkeDvgXWqTlUryoFJsTQrNb6UB-n30tHKrZUxe"

try:
    # Attempt to load service using saved account
    service = QiskitRuntimeService(channel="ibm_cloud")
    print("Account loaded from disk.")
except Exception:
    # Save and load account if not already configured
    QiskitRuntimeService.save_account(channel="ibm_cloud", token=MY_TOKEN, overwrite=True)
    service = QiskitRuntimeService(channel="ibm_cloud")
    print("Account saved and loaded.")

# 2. Define Registers and Circuit
y = QuantumRegister(3, 'y')
x = QuantumRegister(2, 'x')
c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(y, x, c)

# Circuit Operations
qc.x(y[0])            
qc.h(x[1])            
qc.h(x[0])            
qc.cx(y[2], y[0])     
qc.ccx(x[0], y[0], y[2]) 
qc.cx(y[2], y[0])     
qc.h(x[1])
qc.cp(np.pi/2, x[0], x[1])
qc.h(x[0])
qc.swap(x[0], x[1])

# Measurement (x-register only)
qc.measure(x[0], c[0])
qc.measure(x[1], c[1])

# 3. Choose Backend and Transpile
# ibm_fez is selected; ensure you have access to it or modify accordingly
backend = service.backend("ibm_fez") 
tqc = transpile(qc, backend=backend, optimization_level=3)

# 4. Execute 1024 Shots
sampler = Sampler(mode=backend)
job = sampler.run([tqc], shots=1024)

print(f"Job ID: {job.job_id()}")
print("Status: Job is in the queue. Please wait...")

# Process only the x-register bits
result = job.result()
x_counts = result[0].data.c.get_counts()

# Final Visualization
print("Final X-Register Counts:", x_counts)
plot_histogram(x_counts, title=f"Shor's N=15, a=4 (x-register) on {backend.name}")
plt.show()
qc.draw("mpl")
plt.show()
tqc.draw("mpl", cregbundle=True, idle_wires=False)
plt.show()
