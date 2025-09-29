import numpy as np
import matplotlib.pyplot as plt

# Different sample sizes to test
sample_sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for trials in sample_sizes:
    # simulate dice throws
    rolls_a = np.random.randint(1, 7, trials)
    rolls_b = np.random.randint(1, 7, trials)
    total = rolls_a + rolls_b

    # histogram of sums
    freq, bins = np.histogram(total, bins=np.arange(2, 14))

    # plot histogram
    plt.figure(figsize=(6,4))
    plt.bar(bins[:-1], freq / trials, width=0.7, color="skyblue", edgecolor="black")
    plt.title(f"Dice Sum Distribution (n = {trials})")
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Relative Frequency")
    plt.xticks(range(2, 13))
    plt.show()

# We wrote this code to stimulates the rolling of two dices many times 500 upto 1000000.
# After that it adds the results and make for us a histogram of the sums.
# There is also a bar chart which shows how each sum 2-12 happens.
# As the number of rolls getting increasing the bar gets closer to the e probabilities
# As you roll the dice more and more times, the frequencies of each sum (2–12) get closer to their true probabilities.