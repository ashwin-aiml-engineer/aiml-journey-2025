# Day 5: NumPy Fundamentals - Numerical Computing Foundation
import numpy as np
import time
import sys

print("=== NumPy Fundamentals for AI/ML ===")

# NUMPY ARRAY CREATION
print("\n1. NUMPY ARRAY CREATION")
print("-" * 40)

# Different ways to create arrays
arr_from_list = np.array([1, 2, 3, 4, 5])
arr_from_nested = np.array([[1, 2, 3], [4, 5, 6]])
arr_zeros = np.zeros((3, 4))                    # 3x4 array of zeros
arr_ones = np.ones((2, 3))                      # 2x3 array of ones
arr_full = np.full((2, 2), 7)                   # 2x2 array filled with 7
arr_range = np.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
arr_linspace = np.linspace(0, 1, 5)             # 5 equally spaced values from 0 to 1
arr_random = np.random.random((3, 3))           # 3x3 random values [0,1)

print(f"Array from list: {arr_from_list}")
print(f"2D array shape: {arr_from_nested.shape}")
print(f"Zeros array:\n{arr_zeros}")
print(f"Range array: {arr_range}")
print(f"Linspace: {arr_linspace}")

# Array properties
print(f"\nArray properties:")
print(f"Shape: {arr_from_nested.shape}")        # Dimensions
print(f"Size: {arr_from_nested.size}")          # Total elements
print(f"Data type: {arr_from_nested.dtype}")    # Element type
print(f"Dimensions: {arr_from_nested.ndim}")    # Number of dimensions
print(f"Item size: {arr_from_nested.itemsize} bytes")  # Size per element

# ARRAY INDEXING AND SLICING
print("\n2. ARRAY INDEXING AND SLICING")
print("-" * 40)

# 1D array indexing
arr_1d = np.array([10, 20, 30, 40, 50])
print(f"Original 1D array: {arr_1d}")
print(f"First element: {arr_1d[0]}")
print(f"Last element: {arr_1d[-1]}")
print(f"Slice [1:4]: {arr_1d[1:4]}")
print(f"Every 2nd element: {arr_1d[::2]}")

# 2D array indexing
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(f"\n2D array:\n{arr_2d}")
print(f"Element at [1,2]: {arr_2d[1, 2]}")
print(f"First row: {arr_2d[0, :]}")
print(f"Second column: {arr_2d[:, 1]}")
print(f"Sub-array [0:2, 1:3]:\n{arr_2d[0:2, 1:3]}")

# Boolean indexing
print("\nBoolean indexing:")
bool_mask = arr_1d > 25
print(f"Mask (>25): {bool_mask}")
print(f"Elements >25: {arr_1d[bool_mask]}")

# Fancy indexing
indices = np.array([0, 2, 4])
print(f"Elements at indices [0,2,4]: {arr_1d[indices]}")

# ARRAY OPERATIONS AND BROADCASTING
print("\n3. ARRAY OPERATIONS AND BROADCASTING")
print("-" * 40)

# Element-wise operations
arr_a = np.array([1, 2, 3, 4])
arr_b = np.array([10, 20, 30, 40])

print(f"Array A: {arr_a}")
print(f"Array B: {arr_b}")
print(f"A + B: {arr_a + arr_b}")
print(f"A * B: {arr_a * arr_b}")
print(f"A ** 2: {arr_a ** 2}")
print(f"A / B: {arr_a / arr_b}")

# Operations with scalars
print(f"\nScalar operations:")
print(f"A + 10: {arr_a + 10}")
print(f"A * 2: {arr_a * 2}")

# Broadcasting examples
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

print(f"\nBroadcasting:")
print(f"2D array:\n{arr_2d}")
print(f"1D array: {arr_1d}")
print(f"2D + 1D:\n{arr_2d + arr_1d}")

# Different broadcasting scenarios
print(f"\nBroadcasting with different shapes:")
arr_column = np.array([[1], [2]])  # 2x1 array
print(f"Column array shape: {arr_column.shape}")
print(f"2D + Column:\n{arr_2d + arr_column}")

# MATHEMATICAL FUNCTIONS
print("\n4. MATHEMATICAL FUNCTIONS")
print("-" * 40)

# Basic math functions
arr = np.array([1, 4, 9, 16, 25])
print(f"Original array: {arr}")
print(f"Square root: {np.sqrt(arr)}")
print(f"Logarithm: {np.log(arr)}")
print(f"Exponential: {np.exp(np.array([1, 2, 3]))}")

# Trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(f"\nTrigonometric functions:")
print(f"Angles: {angles}")
print(f"Sine: {np.sin(angles)}")
print(f"Cosine: {np.cos(angles)}")

# Aggregation functions
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nAggregation functions:")
print(f"Data:\n{data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")

# Axis-specific operations
print(f"\nAxis-specific operations:")
print(f"Sum along axis 0 (columns): {np.sum(data, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(data, axis=1)}")
print(f"Mean along axis 0: {np.mean(data, axis=0)}")

# ARRAY RESHAPING AND MANIPULATION
print("\n5. ARRAY RESHAPING AND MANIPULATION")
print("-" * 40)

# Reshaping
original = np.arange(12)
print(f"Original array: {original}")
reshaped_2d = original.reshape(3, 4)
print(f"Reshaped (3,4):\n{reshaped_2d}")
reshaped_3d = original.reshape(2, 2, 3)
print(f"Reshaped (2,2,3):\n{reshaped_3d}")

# Flattening
print(f"Flattened back: {reshaped_2d.flatten()}")
print(f"Ravel (view): {reshaped_2d.ravel()}")

# Transposing
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nOriginal matrix:\n{matrix}")
print(f"Transposed:\n{matrix.T}")

# Concatenation and splitting
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"\nConcatenation:")
print(f"Horizontal: {np.concatenate([arr1, arr2])}")
print(f"Vertical: {np.vstack([arr1, arr2])}")
print(f"Horizontal stack: {np.hstack([arr1, arr2])}")

# Splitting
arr_to_split = np.array([1, 2, 3, 4, 5, 6])
split_result = np.split(arr_to_split, 3)
print(f"Split into 3 parts: {split_result}")

# PERFORMANCE COMPARISON: NUMPY VS PYTHON LISTS
print("\n6. PERFORMANCE COMPARISON")
print("-" * 40)

# Create test data
size = 100000
python_list = list(range(size))
numpy_array = np.arange(size)

# Test 1: Sum operation
print(f"Testing with {size} elements:")

# Python list sum
start_time = time.time()
python_sum = sum(python_list)
python_time = time.time() - start_time

# NumPy array sum
start_time = time.time()
numpy_sum = np.sum(numpy_array)
numpy_time = time.time() - start_time

print(f"Python list sum time: {python_time:.6f} seconds")
print(f"NumPy array sum time: {numpy_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster for sum operation")

# Test 2: Element-wise operations
python_list2 = [x * 2 for x in python_list]
start_time = time.time()
python_result = [a + b for a, b in zip(python_list, python_list2)]
python_time = time.time() - start_time

numpy_array2 = numpy_array * 2
start_time = time.time()
numpy_result = numpy_array + numpy_array2
numpy_time = time.time() - start_time

print(f"\nElement-wise addition:")
print(f"Python list time: {python_time:.6f} seconds")
print(f"NumPy array time: {numpy_time:.6f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster for element-wise operations")

# Memory usage comparison
python_memory = sys.getsizeof(python_list)
numpy_memory = numpy_array.nbytes
print(f"\nMemory usage:")
print(f"Python list: {python_memory} bytes")
print(f"NumPy array: {numpy_memory} bytes")
print(f"NumPy uses {python_memory/numpy_memory:.1f}x less memory")

# LINEAR ALGEBRA WITH NUMPY
print("\n7. LINEAR ALGEBRA WITH NUMPY")
print("-" * 40)

# Vector operations
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print(f"Vector A: {vector_a}")
print(f"Vector B: {vector_b}")
print(f"Dot product: {np.dot(vector_a, vector_b)}")
print(f"Cross product: {np.cross(vector_a, vector_b)}")
print(f"Vector magnitude A: {np.linalg.norm(vector_a)}")

# Matrix operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

print(f"\nMatrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")
print(f"Matrix multiplication:\n{np.dot(matrix_a, matrix_b)}")
print(f"Element-wise multiplication:\n{matrix_a * matrix_b}")

# Matrix properties
print(f"Matrix determinant: {np.linalg.det(matrix_a)}")
print(f"Matrix trace: {np.trace(matrix_a)}")

# Solving linear equations: Ax = b
A = np.array([[2, 1], [1, 1]])
b = np.array([3, 2])
x = np.linalg.solve(A, b)
print(f"\nSolving Ax = b:")
print(f"A:\n{A}")
print(f"b: {b}")
print(f"Solution x: {x}")
print(f"Verification Ax: {np.dot(A, x)}")

# RANDOM NUMBER GENERATION
print("\n8. RANDOM NUMBER GENERATION")
print("-" * 40)

# Set seed for reproducibility
np.random.seed(42)

# Different random distributions
uniform_random = np.random.random(5)
normal_random = np.random.normal(0, 1, 5)
integer_random = np.random.randint(1, 100, 5)

print(f"Uniform random [0,1): {uniform_random}")
print(f"Normal distribution: {normal_random}")
print(f"Random integers [1,100): {integer_random}")

# Random sampling
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sample = np.random.choice(data, size=5, replace=False)
print(f"Random sample from data: {sample}")

# Random matrix generation for ML
random_matrix = np.random.randn(3, 4)  # Standard normal distribution
print(f"Random matrix (3x4):\n{random_matrix}")

# PRACTICAL AI/ML EXAMPLES
print("\n9. PRACTICAL AI/ML EXAMPLES")
print("-" * 40)

# Feature scaling/normalization
features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(f"Original features:\n{features}")

# Min-max normalization
min_vals = np.min(features, axis=0)
max_vals = np.max(features, axis=0)
normalized = (features - min_vals) / (max_vals - min_vals)
print(f"Min-max normalized:\n{normalized}")

# Standardization (z-score)
mean_vals = np.mean(features, axis=0)
std_vals = np.std(features, axis=0)
standardized = (features - mean_vals) / std_vals
print(f"Standardized features:\n{standardized}")

# Distance calculations (important for ML)
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])

# Euclidean distance
euclidean_dist = np.sqrt(np.sum((point1 - point2) ** 2))
print(f"\nDistance calculations:")
print(f"Point 1: {point1}")
print(f"Point 2: {point2}")
print(f"Euclidean distance: {euclidean_dist}")
print(f"Manhattan distance: {np.sum(np.abs(point1 - point2))}")

# Correlation coefficient
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 3, 5])
correlation = np.corrcoef(x, y)[0, 1]
print(f"\nCorrelation between x and y: {correlation:.3f}")

# ARRAY BOOLEAN OPERATIONS AND CONDITIONS
print("\n10. BOOLEAN OPERATIONS AND CONDITIONS")
print("-" * 40)

# Boolean arrays
data = np.array([1, 5, 3, 8, 2, 7, 4, 6])
print(f"Data: {data}")

# Boolean conditions
condition1 = data > 4
condition2 = data < 7
combined = condition1 & condition2  # Use & for element-wise AND

print(f"Data > 4: {condition1}")
print(f"Data < 7: {condition2}")
print(f"4 < Data < 7: {combined}")
print(f"Values between 4 and 7: {data[combined]}")

# np.where function
result = np.where(data > 4, data, 0)  # Replace values <=4 with 0
print(f"Replace <=4 with 0: {result}")

# Count and any/all operations
print(f"Count of values > 4: {np.sum(data > 4)}")
print(f"Any value > 8? {np.any(data > 8)}")
print(f"All values > 0? {np.all(data > 0)}")

print("\n=== Day 5 NumPy Fundamentals Completed ===")
print("Key achievements:")
print("✓ Array creation and manipulation")
print("✓ Mathematical operations and broadcasting")
print("✓ Performance advantages over Python lists")
print("✓ Linear algebra operations")
print("✓ Random number generation")
print("✓ Practical AI/ML applications")
print("✓ Boolean operations and filtering")
