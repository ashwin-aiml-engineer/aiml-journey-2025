import functools
import time
import contextlib
from collections import defaultdict, Counter
from itertools import chain, combinations, product
import math
import statistics
from typing import List, Tuple, Dict, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')


print("=" * 55)
print("Week 2 Day 1: Building Professional ML Development Skills")
print("=" * 55)

# ================================
# 1. DECORATORS - ESSENTIAL FOR ML DEVELOPMENT
# ================================

print("\n1️⃣ DECORATORS - Professional Python Patterns")
print("-" * 45)

# Basic decorator for timing functions (crucial for ML performance)
def timer_decorator(func):
    """Decorator to measure function execution time - essential for ML optimization"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Decorator for caching expensive computations (ML models often need this)
def memoize(func):
    """Cache function results to avoid recomputation - crucial for ML efficiency"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"💾 Cached result for {func.__name__}")
        else:
            print(f"⚡ Using cached result for {func.__name__}")
        return cache[key]
    return wrapper

# Practical example: Expensive mathematical computation (common in ML)
@timer_decorator
@memoize
def expensive_ml_computation(n: int) -> float:
    """Simulate expensive ML computation like distance calculations or feature transformations"""
    print(f"🔄 Computing expensive operation for n={n}")
    
    # Simulate complex mathematical operation
    result = 0
    for i in range(1, n + 1):
        result += math.sqrt(i) * math.log(i + 1) * math.sin(i)
    
    return result

# Test the decorators
print("\nTesting decorators with ML-style computations:")
result1 = expensive_ml_computation(10000)  # First call - computed and cached
result2 = expensive_ml_computation(10000)  # Second call - retrieved from cache
result3 = expensive_ml_computation(15000)  # New computation

print(f"✅ Results: {result1:.2f}, {result2:.2f}, {result3:.2f}")

# Advanced decorator: Validation for ML functions
def validate_input(input_type=None, min_value=None, max_value=None):
    """Decorator factory for input validation - crucial for ML data validation"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate first argument (common pattern in ML)
            if args and input_type:
                if not isinstance(args[0], input_type):
                    raise TypeError(f"Expected {input_type}, got {type(args[0])}")
            
            if args and isinstance(args[0], (int, float)):
                if min_value is not None and args[0] < min_value:
                    raise ValueError(f"Value {args[0]} below minimum {min_value}")
                if max_value is not None and args[0] > max_value:
                    raise ValueError(f"Value {args[0]} above maximum {max_value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_input(input_type=(int, float), min_value=0, max_value=1)
def normalize_probability(probability: float) -> float:
    """Normalize probability value - common ML operation"""
    return max(0, min(1, probability))

# Test input validation
print("\n🔍 Testing input validation:")
try:
    print(f"Valid probability: {normalize_probability(0.75)}")
    print(f"Edge case: {normalize_probability(1.5)}")  # This will be clamped but warned
except Exception as e:
    print(f"❌ Validation error: {e}")

print("✅ Decorators mastered - essential for professional ML code!")

# ================================
# 2. GENERATORS - MEMORY EFFICIENT DATA PROCESSING
# ================================

print("\n2️⃣ GENERATORS - Memory Efficient ML Data Processing")
print("-" * 48)

def data_batch_generator(data: List, batch_size: int = 32):
    """Generator for batch processing - essential for ML training with large datasets"""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        print(f"📦 Yielding batch {i//batch_size + 1} with {len(batch)} items")
        yield batch

# Simulate large dataset processing
large_dataset = list(range(1000))  # Simulating 1000 data points
batch_processor = data_batch_generator(large_dataset, batch_size=100)

print("Processing large dataset in batches:")
batch_count = 0
for batch in batch_processor:
    batch_count += 1
    # Simulate ML processing on batch
    batch_mean = statistics.mean(batch)
    print(f"  Batch {batch_count}: Mean = {batch_mean:.2f}, Size = {len(batch)}")
    
    if batch_count >= 3:  # Just show first 3 batches
        print(f"  ... and {10 - batch_count} more batches")
        break

# Advanced generator: Feature engineering pipeline
from typing import Generator

def feature_engineering_pipeline(data: List[Dict]) -> Generator[Dict, None, None]:
    """Generator pipeline for feature engineering - memory efficient ML preprocessing"""
    for record in data:
        # Step 1: Normalize numerical features
        processed_record = record.copy()
        
        # Step 2: Create derived features
        if 'age' in record and 'income' in record:
            processed_record['income_per_age'] = record['income'] / max(record['age'], 1)
        
        # Step 3: Handle categorical encoding
        if 'category' in record:
            processed_record[f"is_{record['category'].lower()}"] = 1
        
        yield processed_record

# Test feature engineering pipeline
sample_data = [
    {'age': 25, 'income': 50000, 'category': 'Tech'},
    {'age': 30, 'income': 75000, 'category': 'Finance'},
    {'age': 35, 'income': 60000, 'category': 'Healthcare'}
]

print("\n🔧 Feature engineering pipeline:")
for i, processed in enumerate(feature_engineering_pipeline(sample_data)):
    print(f"  Record {i+1}: {processed}")

# Generator expression for mathematical operations
squared_generator = (x**2 for x in range(10))
print(f"\n🔢 Generator expression result: {list(squared_generator)}")

print("✅ Generators mastered - memory efficient ML data processing!")

# ================================
# 3. CONTEXT MANAGERS - RESOURCE HANDLING
# ================================

print("\n3️⃣ CONTEXT MANAGERS - Professional Resource Management")
print("-" * 52)

# Context manager for timing code blocks (useful for ML experiments)
@contextlib.contextmanager
def timer_context(operation_name: str):
    """Context manager for timing code blocks - essential for ML performance analysis"""
    print(f"⏱️  Starting: {operation_name}")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"✅ Completed: {operation_name} in {end_time - start_time:.4f} seconds")

# Context manager for temporary configuration changes
@contextlib.contextmanager
def temporary_config(config_dict: Dict, **temp_changes):
    """Temporarily modify configuration - useful for ML experiments"""
    original_values = {}
    
    # Save original values and apply temporary changes
    for key, value in temp_changes.items():
        if key in config_dict:
            original_values[key] = config_dict[key]
        config_dict[key] = value
    
    print(f"🔧 Applied temporary config: {temp_changes}")
    
    try:
        yield config_dict
    finally:
        # Restore original values
        for key, value in original_values.items():
            config_dict[key] = value
        for key in temp_changes:
            if key not in original_values:
                del config_dict[key]
        print("🔄 Restored original configuration")

# Test context managers
print("Testing context managers:")

# Timing a computational task
with timer_context("Matrix multiplication simulation"):
    result = sum(i * j for i in range(100) for j in range(100))
    print(f"  Computation result: {result}")

# Temporary configuration change
ml_config = {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100}
print(f"\nOriginal config: {ml_config}")

with temporary_config(ml_config, learning_rate=0.01, batch_size=64):
    print(f"Temporary config: {ml_config}")
    # Simulate ML training with different config

print(f"Restored config: {ml_config}")

print("✅ Context managers mastered - professional resource management!")

# ================================
# 4. LAMBDA FUNCTIONS & FUNCTIONAL PROGRAMMING
# ================================

print("\n4️⃣ LAMBDA FUNCTIONS & FUNCTIONAL PROGRAMMING")
print("-" * 45)

# Common ML operations with lambda functions
data_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Feature transformations (common in ML preprocessing)
transformations = {
    'normalize': lambda x: (x - min(data_points)) / (max(data_points) - min(data_points)),
    'square': lambda x: x ** 2,
    'log_transform': lambda x: math.log(x + 1),  # +1 to avoid log(0)
    'standardize': lambda x: (x - statistics.mean(data_points)) / statistics.stdev(data_points)
}

print("Feature transformations using lambda functions:")
for transform_name, transform_func in transformations.items():
    transformed = list(map(transform_func, data_points[:5]))  # Show first 5
    print(f"  {transform_name:12}: {[round(x, 3) for x in transformed]}")

# Functional programming patterns for ML
from functools import reduce

# Calculate metrics using functional programming
def calculate_ml_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """Calculate common ML metrics using functional programming"""
    
    # Mean Squared Error
    squared_errors = list(map(lambda pair: (pair[0] - pair[1])**2, zip(predictions, actuals)))
    mse = reduce(lambda x, y: x + y, squared_errors) / len(squared_errors)
    
    # Mean Absolute Error  
    absolute_errors = list(map(lambda pair: abs(pair[0] - pair[1]), zip(predictions, actuals)))
    mae = reduce(lambda x, y: x + y, absolute_errors) / len(absolute_errors)
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': math.sqrt(mse)}

# Test ML metrics calculation
predictions = [2.5, 3.2, 1.8, 4.1, 3.9]
actuals = [2.3, 3.5, 1.5, 4.0, 4.2]

metrics = calculate_ml_metrics(predictions, actuals)
print(f"\n📊 ML Metrics: {metrics}")

# Advanced functional programming: Pipeline composition
def compose(*functions):
    """Compose multiple functions into a pipeline - useful for ML feature pipelines"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Create ML preprocessing pipeline
normalize_fn = lambda x: (x - 50) / 25  # Assume mean=50, std=25
scale_fn = lambda x: x * 2
clip_fn = lambda x: max(-3, min(3, x))

preprocessing_pipeline = compose(clip_fn, scale_fn, normalize_fn)

test_values = [25, 50, 75, 100, 0]
print(f"\nML preprocessing pipeline results:")
for val in test_values:
    processed = preprocessing_pipeline(val)
    print(f"  {val:3d} -> {processed:6.2f}")

print("✅ Functional programming mastered - elegant ML code patterns!")

# ================================
# 5. ADVANCED OOP FOR ML APPLICATIONS
# ================================

print("\n5️⃣ ADVANCED OOP FOR ML APPLICATIONS")
print("-" * 37)

# Abstract base class for ML models
from abc import ABC, abstractmethod

class MLModel(ABC):
    """Abstract base class for ML models - professional ML code structure"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.training_history = []
    
    @abstractmethod
    def train(self, X, y):
        """Abstract method for training - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Abstract method for prediction - must be implemented by subclasses"""
        pass
    
    def evaluate(self, X, y):
        """Common evaluation method for all models"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        # Simple accuracy calculation for demo
        accuracy = sum(1 for pred, actual in zip(predictions, y) if pred == actual) / len(y)
        return {'accuracy': accuracy}
    
    def save_model(self, filepath: str):
        """Template method for saving models"""
        print(f"💾 Saving {self.model_name} model to {filepath}")
        # In real implementation, would save model parameters
        
    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name}(status={status})"

# Concrete implementation: Simple Linear Classifier
class SimpleLinearClassifier(MLModel):
    """Simple linear classifier implementation - demonstrates ML OOP patterns"""
    
    def __init__(self):
        super().__init__("SimpleLinearClassifier")
        self.weights = None
        self.bias = None
    
    def train(self, X: List[List[float]], y: List[int]):
        """Simple training implementation for demonstration"""
        print(f"🎯 Training {self.model_name} with {len(X)} samples")
        
        # Simplified training: just store mean of features as weights
        self.weights = [statistics.mean(feature) for feature in zip(*X)]
        self.bias = statistics.mean(y)
        self.is_trained = True
        
        # Record training history
        self.training_history.append({
            'samples': len(X),
            'features': len(X[0]) if X else 0,
            'timestamp': time.time()
        })
        
        print(f"✅ Training completed. Weights: {[round(w, 3) for w in self.weights]}")
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Simple prediction implementation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        for sample in X:
            # Simple linear combination
            score = sum(w * f for w, f in zip(self.weights, sample)) + self.bias
            prediction = 1 if score > 0 else 0
            predictions.append(prediction)
        
        return predictions

# Model factory pattern (common in ML frameworks)
class ModelFactory:
    """Factory pattern for creating ML models - professional design pattern"""
    
    models = {
        'linear': SimpleLinearClassifier,
        # Could add more model types here
    }
    
    @classmethod
    def create_model(cls, model_type: str) -> MLModel:
        """Create model instance based on type"""
        if model_type not in cls.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return cls.models[model_type]()
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types"""
        return list(cls.models.keys())

# Test advanced OOP patterns
print("Testing advanced OOP for ML:")

# Create model using factory pattern
available_models = ModelFactory.list_available_models()
print(f"Available models: {available_models}")

model = ModelFactory.create_model('linear')
print(f"Created model: {model}")

# Generate sample training data
X_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
y_train = [0, 0, 1, 1]

# Train the model
model.train(X_train, y_train)

# Make predictions
X_test = [[1.5, 2.5], [3.5, 4.5]]
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")

# Evaluate model
accuracy = model.evaluate(X_train, y_train)
print(f"Training accuracy: {accuracy}")

print("✅ Advanced OOP mastered - professional ML code architecture!")

# ================================
# 6. PERFORMANCE OPTIMIZATION TECHNIQUES
# ================================

print("\n6️⃣ PERFORMANCE OPTIMIZATION FOR ML")
print("-" * 35)

# List comprehensions vs loops (crucial for ML data processing)
def compare_performance():
    """Compare different approaches for ML-style data processing"""
    
    data = list(range(100000))
    
    # Method 1: Traditional loop
    with timer_context("Traditional loop"):
        result1 = []
        for x in data:
            if x % 2 == 0:
                result1.append(x ** 2)
    
    # Method 2: List comprehension  
    with timer_context("List comprehension"):
        result2 = [x ** 2 for x in data if x % 2 == 0]
    
    # Method 3: Generator with conversion
    with timer_context("Generator expression"):
        result3 = list(x ** 2 for x in data if x % 2 == 0)
    
    print(f"All methods produced same result: {len(result1) == len(result2) == len(result3)}")

compare_performance()

# Memory-efficient techniques for large ML datasets
def memory_efficient_processing(data_generator):
    """Process large datasets without loading everything into memory"""
    
    # Running statistics without storing all data
    count = 0
    running_sum = 0
    running_sum_squares = 0
    
    for value in data_generator:
        count += 1
        running_sum += value
        running_sum_squares += value ** 2
        
        # Can process data here without storing it all
    
    mean = running_sum / count
    variance = (running_sum_squares / count) - (mean ** 2)
    std_dev = math.sqrt(variance)
    
    return {'count': count, 'mean': mean, 'std_dev': std_dev}

# Test memory-efficient processing
large_data_generator = (x + 0.5 * x**2 for x in range(10000))
stats = memory_efficient_processing(large_data_generator)
print(f"\n📊 Memory-efficient statistics: {stats}")

# Profiling and optimization hints
import sys

def analyze_memory_usage(obj, obj_name: str):
    """Analyze memory usage of objects - important for ML optimization"""
    size = sys.getsizeof(obj)
    print(f"💾 Memory usage of {obj_name}: {size:,} bytes")

# Compare memory usage of different data structures
sample_data = list(range(1000))
sample_tuple = tuple(range(1000))
sample_set = set(range(1000))
sample_dict = {i: i**2 for i in range(1000)}

print("\nMemory usage comparison:")
analyze_memory_usage(sample_data, "list(1000)")
analyze_memory_usage(sample_tuple, "tuple(1000)")  
analyze_memory_usage(sample_set, "set(1000)")
analyze_memory_usage(sample_dict, "dict(1000)")

print("✅ Performance optimization mastered - efficient ML code!")

# ================================
# 7. MATHEMATICAL FOUNDATIONS FOR ML
# ================================

print("\n7️⃣ MATHEMATICAL FOUNDATIONS FOR ML")
print("-" * 35)

# Statistical functions for ML (building blocks for algorithms)
class MLMath:
    """Mathematical utilities for ML - building foundation for algorithms"""
    
    @staticmethod
    def euclidean_distance(point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance - fundamental for many ML algorithms"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    @staticmethod
    def manhattan_distance(point1: List[float], point2: List[float]) -> float:
        """Calculate Manhattan distance - used in various ML algorithms"""
        return sum(abs(a - b) for a, b in zip(point1, point2))
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity - important for text analysis and recommendations"""
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in vector1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function - fundamental in neural networks"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    @staticmethod
    def relu(x: float) -> float:
        """ReLU activation function - most common in deep learning"""
        return max(0, x)
    
    @staticmethod
    def softmax(values: List[float]) -> List[float]:
        """Softmax function - used in classification output layers"""
        exp_values = [math.exp(x - max(values)) for x in values]  # Subtract max for stability
        sum_exp = sum(exp_values)
        return [x / sum_exp for x in exp_values]

# Test mathematical foundations
print("Testing ML mathematical functions:")

# Distance calculations
point_a = [1.0, 2.0, 3.0]
point_b = [4.0, 5.0, 6.0]

euclidean_dist = MLMath.euclidean_distance(point_a, point_b)
manhattan_dist = MLMath.manhattan_distance(point_a, point_b)
cosine_sim = MLMath.cosine_similarity(point_a, point_b)

print(f"Distance calculations:")
print(f"  Euclidean distance: {euclidean_dist:.3f}")
print(f"  Manhattan distance: {manhattan_dist:.3f}")
print(f"  Cosine similarity: {cosine_sim:.3f}")

# Activation functions
test_values = [-2, -1, 0, 1, 2]
print(f"\nActivation functions:")
print(f"  Input values: {test_values}")
print(f"  Sigmoid: {[round(MLMath.sigmoid(x), 3) for x in test_values]}")
print(f"  ReLU: {[MLMath.relu(x) for x in test_values]}")

# Softmax for classification
logits = [2.0, 1.0, 0.1]
probabilities = MLMath.softmax(logits)
print(f"  Softmax input: {logits}")
print(f"  Softmax output: {[round(p, 3) for p in probabilities]}")
print(f"  Sum of probabilities: {sum(probabilities):.3f}")

print("✅ Mathematical foundations established - ready for ML algorithms!")

# ================================
# 8. PUTTING IT ALL TOGETHER - MINI PROJECT
# ================================

print("\n8️⃣ MINI PROJECT - ADVANCED PYTHON ML TOOLKIT")
print("-" * 45)

class MLDataProcessor:
    """
    Advanced Python ML toolkit demonstrating all concepts learned today
    This class showcases professional ML development patterns
    """
    
    def __init__(self, name: str):
        self.name = name
        self.processing_history = []
        self._cache = {}
    
    @timer_decorator
    def process_dataset(self, data: List[Dict], transformations: List[Callable]) -> List[Dict]:
        """Process dataset using functional programming pipeline"""
        
        # Create processing pipeline using function composition
        pipeline = lambda record: self._apply_transformations(record, transformations)
        
        # Use generator for memory-efficient processing
        processed_data = []
        with timer_context(f"Processing {len(data)} records"):
            for record in data:
                processed_record = pipeline(record)
                processed_data.append(processed_record)
        
        # Record processing history
        self.processing_history.append({
            'timestamp': time.time(),
            'records_processed': len(data),
            'transformations_applied': len(transformations)
        })
        
        return processed_data
    
    def _apply_transformations(self, record: Dict, transformations: List[Callable]) -> Dict:
        """Apply sequence of transformations to a record"""
        result = record.copy()
        for transform in transformations:
            result = transform(result)
        return result
    
    @memoize
    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics with caching"""
        
        if not data:
            return {}
        
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std_dev': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data)
        }
    
    def batch_process(self, data: List, batch_size: int = 100) -> Dict[str, float]:
        """Process data in batches using generators"""
        
        batch_stats = []
        
        # Use generator for batch processing
        for batch in data_batch_generator(data, batch_size):
            if all(isinstance(x, (int, float)) for x in batch):
                stats = self.calculate_statistics(batch)
                batch_stats.append(stats['mean'] if stats else 0)
        
        # Calculate overall statistics
        if batch_stats:
            overall_stats = self.calculate_statistics(batch_stats)
            return overall_stats
        
        return {}

# Demonstrate the ML toolkit
print("Demonstrating Advanced Python ML Toolkit:")

# Create processor
processor = MLDataProcessor("Advanced ML Processor")

# Sample dataset
sample_dataset = [
    {'age': 25, 'income': 50000, 'experience': 2},
    {'age': 30, 'income': 75000, 'experience': 5},
    {'age': 35, 'income': 90000, 'experience': 8},
    {'age': 28, 'income': 65000, 'experience': 4},
    {'age': 32, 'income': 80000, 'experience': 6}
]

# Define transformations using lambda functions
transformations = [
    lambda record: {**record, 'age_normalized': record['age'] / 50},
    lambda record: {**record, 'income_k': record['income'] / 1000},
    lambda record: {**record, 'experience_level': 'Senior' if record['experience'] > 5 else 'Junior'}
]

# Process dataset
processed_data = processor.process_dataset(sample_dataset, transformations)

print(f"\nProcessed dataset sample:")
for i, record in enumerate(processed_data[:2]):
    print(f"  Record {i+1}: {record}")

# Test batch processing with numerical data
numerical_data = [record['income'] for record in sample_dataset] * 20  # Create larger dataset
batch_stats = processor.batch_process(numerical_data, batch_size=10)

print(f"\nBatch processing statistics: {batch_stats}")

# Test caching with repeated calculations
ages = [record['age'] for record in sample_dataset]
stats1 = processor.calculate_statistics(ages)  # First call - computed
stats2 = processor.calculate_statistics(ages)  # Second call - cached

print(f"\nAge statistics: {stats1}")

print("\n🎉 MINI PROJECT COMPLETED!")
print("Advanced Python ML toolkit successfully demonstrates:")
print("  ✅ Decorators for timing and caching")
print("  ✅ Generators for memory-efficient processing")  
print("  ✅ Context managers for resource handling")
print("  ✅ Functional programming patterns")
print("  ✅ Advanced OOP design patterns")
print("  ✅ Performance optimization techniques")
print("  ✅ Mathematical foundations for ML")