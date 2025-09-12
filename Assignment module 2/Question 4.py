import numpy as np

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)

A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\nInverse of A (A^-1):")
print(A_inv)

print("\nA * A^-1:")
print(A @ A_inv)

print("\nA^-1 * A:")
print(A_inv @ A)
