import csv
import numpy as np
import matplotlib.pyplot as plt

h = []
w = []

with open("weight-height.csv", "r") as f:
    data = csv.DictReader(f)
    for row in data:
        h.append(float(row["Height"]))
        w.append(float(row["Weight"]))

h = np.array(h)
w = np.array(w)

table = np.column_stack((h, w))

h_cm = h * 2.54
w_kg = w * 0.45359237

print("Average height (cm):", round(np.mean(h_cm), 2))
print("Average weight (kg):", round(np.mean(w_kg), 2))

plt.hist(h_cm, bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram of Heights (cm)")
plt.xlabel("Height (cm)")
plt.ylabel("Count")
plt.show()
