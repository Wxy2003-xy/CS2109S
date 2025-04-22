import matplotlib.pyplot as plt
import numpy as np
from utils import load_data

# Load data
data = load_data()

# Save data to txt
with open("data_dump.txt", "w") as f:
    for i, (state, utility) in enumerate(data):
        f.write(f"--- Entry {i} ---\n")
        f.write(f"Utility: {utility}\n")
        f.write(f"State:\n{state}\n\n")

# Extract utility values
utilities = np.array([utility for _, utility in data])

# Plot histogram
plt.hist(utilities, bins=50, alpha=0.7, edgecolor='black')
plt.title("Utility Value Distribution")
plt.xlabel("Utility")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
