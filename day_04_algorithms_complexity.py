# Day 4: Basic Algorithms & Complexity Analysis
print("=== Basic Algorithms & Complexity Analysis ===")

import time
import random

# SEARCHING ALGORITHMS
print("\n1. SEARCHING ALGORITHMS")
print("-" * 40)

def linear_search(arr, target):
    """Linear search - O(n) time complexity"""
    comparisons = 0
    for i, element in enumerate(arr):
        comparisons += 1
        if element == target:
            return i, comparisons
    return -1, comparisons

def binary_search(arr, target):
    """Binary search - O(log n) time complexity - requires sorted array"""
    left, right = 0, len(arr) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons

# Test searching algorithms
test_data = list(range(1, 1001))  # Sorted list 1-1000
target = 750

# Linear search
linear_result, linear_comparisons = linear_search(test_data, target)
print(f"Linear search: Found {target} at index {linear_result} in {linear_comparisons} comparisons")

# Binary search
binary_result, binary_comparisons = binary_search(test_data, target)
print(f"Binary search: Found {target} at index {binary_result} in {binary_comparisons} comparisons")

print(f"Binary search is {linear_comparisons/binary_comparisons:.1f}x more efficient!")

# SORTING ALGORITHMS
print("\n2. SORTING ALGORITHMS")
print("-" * 40)

def bubble_sort(arr):
    """Bubble sort - O(n²) time complexity"""
    arr = arr.copy()  # Don't modify original
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
    
    return arr, comparisons, swaps

def selection_sort(arr):
    """Selection sort - O(n²) time complexity"""
    arr = arr.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1
    
    return arr, comparisons, swaps

def insertion_sort(arr):
    """Insertion sort - O(n²) time complexity but efficient for small arrays"""
    arr = arr.copy()
    comparisons = 0
    shifts = 0
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        while j >= 0:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                shifts += 1
            else:
                break
        
        arr[j + 1] = key
    
    return arr, comparisons, shifts

def merge_sort(arr):
    """Merge sort - O(n log n) time complexity"""
    if len(arr) <= 1:
        return arr, 0
    
    mid = len(arr) // 2
    left, left_ops = merge_sort(arr[:mid])
    right, right_ops = merge_sort(arr[mid:])
    
    merged, merge_ops = merge(left, right)
    total_ops = left_ops + right_ops + merge_ops
    
    return merged, total_ops

def merge(left, right):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    operations = 0
    
    while i < len(left) and j < len(right):
        operations += 1
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result, operations

def quick_sort(arr):
    """Quick sort - O(n log n) average, O(n²) worst case"""
    if len(arr) <= 1:
        return arr, 0
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    left_sorted, left_ops = quick_sort(left)
    right_sorted, right_ops = quick_sort(right)
    
    operations = left_ops + right_ops + len(arr)  # Partitioning cost
    
    return left_sorted + middle + right_sorted, operations

# Test sorting algorithms
test_sizes = [50, 100, 200]
algorithms = [
    ("Bubble Sort", bubble_sort),
    ("Selection Sort", selection_sort),
    ("Insertion Sort", insertion_sort),
    ("Merge Sort", merge_sort),
    ("Quick Sort", quick_sort)
]

print("Algorithm Performance Comparison:")
print("Size\t", end="")
for name, _ in algorithms:
    print(f"{name}\t", end="")
print()

for size in test_sizes:
    test_array = random.sample(range(1000), size)
    print(f"{size}\t", end="")
    
    for name, algorithm in algorithms:
        start_time = time.time()
        result = algorithm(test_array)
        if len(result) == 2:
            sorted_arr, operations = result
        else:
            sorted_arr, operations, _ = result  # Ignore third value
        end_time = time.time()
        
        print(f"{operations}\t\t", end="")
    print()

# BIG O NOTATION EXAMPLES
print("\n3. BIG O NOTATION EXAMPLES")
print("-" * 40)

def demonstrate_complexity():
    """Demonstrate different time complexities"""
    
    # O(1) - Constant time
    def constant_time_operation(arr):
        """Always takes same time regardless of input size"""
        return arr[0] if arr else None
    
    # O(n) - Linear time
    def linear_time_operation(arr):
        """Time increases linearly with input size"""
        return sum(arr)
    
    # O(n²) - Quadratic time
    def quadratic_time_operation(arr):
        """Time increases quadratically with input size"""
        result = 0
        for i in arr:
            for j in arr:
                result += i * j
        return result
    
    # O(log n) - Logarithmic time
    def logarithmic_time_operation(n):
        """Time increases logarithmically"""
        count = 0
        while n > 1:
            n //= 2
            count += 1
        return count
    
    # Test with different sizes
    sizes = [100, 1000, 2000]
    
    for size in sizes:
        test_data = list(range(size))
        
        # O(1)
        start = time.time()
        constant_time_operation(test_data)
        constant_time = time.time() - start
        
        # O(n)
        start = time.time()
        linear_time_operation(test_data)
        linear_time = time.time() - start
        
        # O(log n)
        start = time.time()
        logarithmic_time_operation(size)
        log_time = time.time() - start
        
        print(f"Size {size}: O(1)={constant_time:.6f}s, O(n)={linear_time:.6f}s, O(log n)={log_time:.6f}s")

demonstrate_complexity()

# ALGORITHM OPTIMIZATION EXAMPLES
print("\n4. ALGORITHM OPTIMIZATION EXAMPLES")
print("-" * 40)

# Finding duplicates - multiple approaches
def find_duplicates_naive(arr):
    """Naive approach - O(n²)"""
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates

def find_duplicates_optimized(arr):
    """Optimized approach using set - O(n)"""
    seen = set()
    duplicates = set()
    
    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)

def find_duplicates_counter(arr):
    """Using Counter from collections - O(n)"""
    from collections import Counter
    counts = Counter(arr)
    return [item for item, count in counts.items() if count > 1]

# Test duplicate finding
test_array = [1, 2, 3, 4, 2, 5, 6, 3, 7, 8, 1, 9] * 100  # Larger test

print("Finding duplicates performance:")
start_time = time.time()
duplicates1 = find_duplicates_naive(test_array[:100])  # Smaller for naive
naive_time = time.time() - start_time

start_time = time.time()
duplicates2 = find_duplicates_optimized(test_array)
optimized_time = time.time() - start_time

start_time = time.time()
duplicates3 = find_duplicates_counter(test_array)
counter_time = time.time() - start_time

print(f"Naive approach: {naive_time:.6f}s")
print(f"Optimized approach: {optimized_time:.6f}s")
print(f"Counter approach: {counter_time:.6f}s")

# RECURSIVE ALGORITHMS
print("\n5. RECURSIVE ALGORITHMS")
print("-" * 40)

def factorial_recursive(n):
    """Factorial using recursion"""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

def factorial_iterative(n):
    """Factorial using iteration"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def fibonacci_recursive(n):
    """Fibonacci using recursion - inefficient"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_iterative(n):
    """Fibonacci using iteration - efficient"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_memoized(n, memo={}):
    """Fibonacci with memoization - efficient recursion"""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

# Test recursive vs iterative
print("Factorial comparison:")
n = 20
print(f"Recursive factorial({n}): {factorial_recursive(n)}")
print(f"Iterative factorial({n}): {factorial_iterative(n)}")

print("\nFibonacci comparison (n=30):")
n = 30

start_time = time.time()
fib_iter = fibonacci_iterative(n)
iter_time = time.time() - start_time

start_time = time.time()
fib_memo = fibonacci_memoized(n)
memo_time = time.time() - start_time

print(f"Iterative: {fib_iter} in {iter_time:.6f}s")
print(f"Memoized: {fib_memo} in {memo_time:.6f}s")

# Don't test recursive fibonacci for n=30 as it's too slow!
print("Recursive fibonacci is too slow for n=30!")

# DATA STRUCTURE ALGORITHMS
print("\n6. DATA STRUCTURE ALGORITHMS")
print("-" * 40)

# Stack implementation and algorithms
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop() if self.items else None
    
    def peek(self):
        return self.items[-1] if self.items else None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

def balanced_parentheses(expression):
    """Check if parentheses are balanced using stack"""
    stack = Stack()
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in expression:
        if char in pairs:
            stack.push(char)
        elif char in pairs.values():
            if stack.is_empty():
                return False
            if pairs[stack.pop()] != char:
                return False
    
    return stack.is_empty()

# Test balanced parentheses
test_expressions = [
    "((()))",
    "([{}])",
    "(()",
    "([)]",
    "{[()()]}"
]

print("Balanced parentheses check:")
for expr in test_expressions:
    result = balanced_parentheses(expr)
    print(f"{expr}: {'Balanced' if result else 'Not balanced'}")

# Queue implementation using two stacks
class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue
    
    def enqueue(self, item):
        self.stack1.append(item)
    
    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        return self.stack2.pop() if self.stack2 else None

# Test queue implementation
queue = QueueUsingStacks()
for i in range(5):
    queue.enqueue(f"Task_{i}")

print("\nQueue operations:")
while True:
    item = queue.dequeue()
    if item is None:
        break
    print(f"Dequeued: {item}")

# ALGORITHMIC PROBLEM SOLVING
print("\n7. ALGORITHMIC PROBLEM SOLVING")
print("-" * 40)

def two_sum(nums, target):
    """Find two numbers that add up to target - O(n) using hash map"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def max_subarray_sum(arr):
    """Kadane's algorithm for maximum subarray sum - O(n)"""
    max_sum = current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def longest_common_prefix(strs):
    """Find longest common prefix among strings"""
    if not strs:
        return ""
    
    prefix = strs[0]
    for string in strs[1:]:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

# Test problem-solving algorithms
print("Two sum problem:")
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(f"Array: {nums}, Target: {target}")
print(f"Indices: {result} -> Values: [{nums[result[0]]}, {nums[result[1]]}]")

print("\nMaximum subarray sum:")
test_array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = max_subarray_sum(test_array)
print(f"Array: {test_array}")
print(f"Maximum subarray sum: {max_sum}")

print("\nLongest common prefix:")
strings = ["flower", "flow", "flight"]
prefix = longest_common_prefix(strings)
print(f"Strings: {strings}")
print(f"Longest common prefix: '{prefix}'")

# ALGORITHM ANALYSIS SUMMARY
print("\n8. ALGORITHM ANALYSIS SUMMARY")
print("-" * 40)

complexity_guide = {
    "O(1)":
    ["Array access", "Hash table lookup", "Stack push/pop"],
    "O(log n)": ["Binary search", "Binary tree operations", "Heap operations"],
    "O(n)": ["Linear search", "Array traversal", "Finding min/max"],
    "O(n log n)": ["Merge sort", "Quick sort (average)", "Heap sort"],
    "O(n²)": ["Bubble sort", "Selection sort", "Nested loops"],
    "O(2^n)": ["Recursive fibonacci", "Tower of Hanoi", "Subset generation"]
}

print("Time Complexity Guide:")
for complexity, examples in complexity_guide.items():
    print(f"{complexity:8}: {', '.join(examples)}")

space_complexity_guide = {
    "O(1)": "Iterative algorithms, in-place sorting",
    "O(n)": "Recursive calls, auxiliary arrays",
    "O(n²)": "2D matrices, nested data structures"
}

print("\nSpace Complexity Guide:")
for complexity, description in space_complexity_guide.items():
    print(f"{complexity:8}: {description}")

# PRACTICAL ML ALGORITHM EXAMPLES
print("\n9. PRACTICAL ML ALGORITHM EXAMPLES")
print("-" * 40)

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def k_nearest_neighbors(data, query_point, k=3):
    """Simple k-NN implementation"""
    # Calculate distances to all points
    distances = []
    for point, label in data:
        dist = euclidean_distance(point, query_point)
        distances.append((dist, label))
    
    # Sort by distance and take k nearest
    distances.sort()
    k_nearest = distances[:k]
    
    # Return most common label
    labels = [label for _, label in k_nearest]
    return max(set(labels), key=labels.count)

# Test k-NN
training_data = [
    ([1, 2], "A"), ([2, 3], "A"), ([3, 3], "A"),
    ([6, 5], "B"), ([7, 7], "B"), ([8, 6], "B")
]

test_point = [4, 4]
prediction = k_nearest_neighbors(training_data, test_point, k=3)
print(f"k-NN prediction for point {test_point}: {prediction}")

def linear_regression_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Simple linear regression using gradient descent"""
    m = len(X)
    theta = 0.0  # Parameter to learn
    
    for _ in range(epochs):
        # Forward pass
        predictions = [theta * x for x in X]
        
        # Calculate cost (MSE)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / (2 * m)
        
        # Calculate gradient
        gradient = sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X)) / m
        
        # Update parameter
        theta -= learning_rate * gradient
    
    return theta

# Test linear regression
X_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]  # Perfect linear relationship: y = 2x

learned_theta = linear_regression_gradient_descent(X_train, y_train)
print(f"Learned parameter: {learned_theta:.3f} (expected: 2.0)")

# ALGORITHM SELECTION GUIDE
print("\n10. ALGORITHM SELECTION GUIDE")
print("-" * 40)

selection_guide = {
    "Small dataset (<100 items)": {
        "Sorting": "Insertion sort",
        "Searching": "Linear search", 
        "Reason": "Simple implementation, low overhead"
    },
    "Medium dataset (100-10000 items)": {
        "Sorting": "Quick sort or Merge sort",
        "Searching": "Binary search (if sorted)",
        "Reason": "Good balance of speed and memory"
    },
    "Large dataset (>10000 items)": {
        "Sorting": "Merge sort (stable) or Heap sort",
        "Searching": "Hash table or Binary search tree",
        "Reason": "Guaranteed performance, scalability"
    },
    "Memory-constrained": {
        "Sorting": "Heap sort (O(1) space)",
        "Searching": "Binary search",
        "Reason": "Minimal extra memory usage"
    },
    "Real-time requirements": {
        "Sorting": "Quick sort (average case)",
        "Searching": "Hash table",
        "Reason": "Fast average performance"
    }
}

for scenario, recommendations in selection_guide.items():
    print(f"\n{scenario}:")
    for operation, algorithm in recommendations.items():
        if operation != "Reason":
            print(f"  {operation}: {algorithm}")
    print(f"  Reason: {recommendations['Reason']}")

print("\nDay 4 Algorithms and Complexity Analysis completed!")
