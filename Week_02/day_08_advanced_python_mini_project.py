import time
import functools
import contextlib
import math
import statistics
import json
from typing import List, Dict, Callable, Optional, Union, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 ADVANCED PYTHON ML DATA PIPELINE")
print("=" * 50)
print("Professional-grade data processing system")
print("=" * 50)

# ================================
# 1. DECORATORS FOR MONITORING AND CACHING
# ================================

def performance_monitor(func):
    """Decorator to monitor function performance - essential for ML pipelines"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0  # Simplified - in production would use memory profiling
        
        print(f"🔄 Starting: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            print(f"✅ Completed: {func.__name__}")
            print(f"   ⏱️  Execution time: {execution_time:.4f}s")
            print(f"   📊 Records processed: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

def intelligent_cache(max_size: int = 128):
    """Advanced caching decorator with size limit"""
    def decorator(func):
        cache = {}
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                # Move to end (most recently used)
                access_order.remove(key)
                access_order.append(key)
                print(f"💾 Cache hit for {func.__name__}")
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            access_order.append(key)
            
            # Evict least recently used if cache is full
            if len(cache) > max_size:
                lru_key = access_order.pop(0)
                del cache[lru_key]
                print(f"🗑️  Evicted LRU item from cache")
            
            print(f"💾 Cached result for {func.__name__}")
            return result
        
        wrapper.cache_info = lambda: {
            'cache_size': len(cache),
            'max_size': max_size,
            'hit_ratio': 'N/A'  # Simplified
        }
        
        return wrapper
    return decorator

# ================================
# 2. DATA CLASSES AND TYPE DEFINITIONS
# ================================

@dataclass
class DataRecord:
    """Structured data record for ML processing"""
    id: int
    features: Dict[str, Union[float, str, int]]
    target: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {'created_at': datetime.now().isoformat()}

@dataclass
class ProcessingStats:
    """Statistics for pipeline processing"""
    records_processed: int
    processing_time: float
    errors_encountered: int
    memory_efficiency: float
    transformations_applied: int

# ================================
# 3. ADVANCED DATA GENERATORS
# ================================

def data_stream_generator(data_source: List[Dict], batch_size: int = 100) -> Iterator[List[DataRecord]]:
    """
    Memory-efficient data streaming generator
    Simulates reading from large data sources without loading everything into memory
    """
    print(f"📡 Initializing data stream with batch size: {batch_size}")
    
    current_batch = []
    
    for i, raw_record in enumerate(data_source):
        # Convert raw data to structured record
        record = DataRecord(
            id=raw_record.get('id', i),
            features=raw_record.get('features', {}),
            target=raw_record.get('target'),
            metadata={'batch_index': i // batch_size}
        )
        
        current_batch.append(record)
        
        # Yield batch when full
        if len(current_batch) >= batch_size:
            print(f"📦 Yielding batch {i // batch_size + 1} with {len(current_batch)} records")
            yield current_batch
            current_batch = []
    
    # Yield remaining records
    if current_batch:
        print(f"📦 Yielding final batch with {len(current_batch)} records")
        yield current_batch

def feature_engineering_generator(records: Iterator[DataRecord]) -> Iterator[DataRecord]:
    """
    Generator for feature engineering transformations
    Applies transformations without loading all data into memory
    """
    for record in records:
        # Apply feature engineering
        enhanced_features = record.features.copy()
        
        # Example transformations
        if 'age' in enhanced_features and 'income' in enhanced_features:
            enhanced_features['income_per_age'] = enhanced_features['income'] / max(enhanced_features['age'], 1)
        
        if 'experience' in enhanced_features:
            enhanced_features['experience_level'] = (
                'Senior' if enhanced_features['experience'] > 5 
                else 'Mid' if enhanced_features['experience'] > 2 
                else 'Junior'
            )
        
        # Create enhanced record
        enhanced_record = DataRecord(
            id=record.id,
            features=enhanced_features,
            target=record.target,
            metadata={**record.metadata, 'feature_engineered': True}
        )
        
        yield enhanced_record

# ================================
# 4. CONTEXT MANAGERS FOR RESOURCE HANDLING
# ================================

@contextlib.contextmanager
def pipeline_execution_context(pipeline_name: str, config: Dict):
    """Context manager for ML pipeline execution with automatic cleanup"""
    
    print(f"🚀 Starting pipeline: {pipeline_name}")
    print(f"📋 Configuration: {config}")
    
    start_time = time.time()
    errors = []
    
    try:
        # Setup phase
        yield {
            'pipeline_name': pipeline_name,
            'start_time': start_time,
            'errors': errors
        }
        
    except Exception as e:
        errors.append(str(e))
        print(f"❌ Pipeline error: {e}")
        raise
        
    finally:
        # Cleanup and reporting phase
        execution_time = time.time() - start_time
        status = "SUCCESS" if not errors else "FAILED"
        
        print(f"📊 Pipeline: {pipeline_name}")
        print(f"   Status: {status}")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Errors: {len(errors)}")
        
        if errors:
            print(f"   Error details: {errors}")

@contextlib.contextmanager
def temporary_config_override(config_dict: Dict, **overrides):
    """Temporarily override configuration for experimentation"""
    
    original_values = {}
    
    # Save original values and apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            original_values[key] = config_dict[key]
        config_dict[key] = value
    
    print(f"⚙️  Applied config overrides: {overrides}")
    
    try:
        yield config_dict
    finally:
        # Restore original configuration
        for key, value in original_values.items():
            config_dict[key] = value
        
        for key in overrides:
            if key not in original_values:
                del config_dict[key]
        
        print(f"🔄 Restored original configuration")

# ================================
# 5. ADVANCED ML DATA PROCESSOR CLASS
# ================================

class AdvancedMLDataProcessor(ABC):
    """Abstract base class for ML data processors"""
    
    def __init__(self, processor_name: str):
        self.processor_name = processor_name
        self.processing_history = []
        self.config = {
            'batch_size': 100,
            'enable_caching': True,
            'performance_monitoring': True,
            'error_tolerance': 0.05
        }
    
    @abstractmethod
    def process_batch(self, batch: List[DataRecord]) -> List[DataRecord]:
        """Process a batch of records - must be implemented by subclasses"""
        pass
    
    @performance_monitor
    def process_dataset(self, data_source: List[Dict]) -> List[DataRecord]:
        """Process entire dataset using advanced Python techniques"""
        
        config = self.config
        all_processed_records = []
        total_errors = 0
        
        with pipeline_execution_context(f"{self.processor_name}_pipeline", config) as context:
            
            # Stream data in batches (memory efficient)
            data_stream = data_stream_generator(data_source, config['batch_size'])
            
            # Process each batch
            for batch_num, batch in enumerate(data_stream):
                try:
                    # Apply feature engineering
                    engineered_batch = list(feature_engineering_generator(iter(batch)))
                    
                    # Process batch using subclass implementation
                    processed_batch = self.process_batch(engineered_batch)
                    
                    all_processed_records.extend(processed_batch)
                    
                except Exception as e:
                    total_errors += 1
                    context['errors'].append(f"Batch {batch_num}: {str(e)}")
                    
                    # Check error tolerance
                    error_rate = total_errors / (batch_num + 1)
                    if error_rate > config['error_tolerance']:
                        raise RuntimeError(f"Error rate {error_rate:.2%} exceeds tolerance")
        
        # Record processing statistics
        stats = ProcessingStats(
            records_processed=len(all_processed_records),
            processing_time=time.time(),
            errors_encountered=total_errors,
            memory_efficiency=1.0,  # Simplified
            transformations_applied=2  # Feature engineering + custom processing
        )
        
        self.processing_history.append(stats)
        
        return all_processed_records
    
    def get_processing_summary(self) -> Dict:
        """Get summary of all processing operations"""
        if not self.processing_history:
            return {'message': 'No processing history available'}
        
        latest_stats = self.processing_history[-1]
        
        return {
            'processor_name': self.processor_name,
            'total_runs': len(self.processing_history),
            'latest_stats': {
                'records_processed': latest_stats.records_processed,
                'errors_encountered': latest_stats.errors_encountered,
                'transformations_applied': latest_stats.transformations_applied
            },
            'configuration': self.config
        }

class SmartMLDataProcessor(AdvancedMLDataProcessor):
    """Concrete implementation of advanced ML data processor"""
    
    def __init__(self):
        super().__init__("SmartMLDataProcessor")
        self.transformation_functions = []
        self.validation_rules = []
    
    def add_transformation(self, transform_func: Callable[[DataRecord], DataRecord]):
        """Add custom transformation function"""
        self.transformation_functions.append(transform_func)
        print(f"➕ Added transformation function: {transform_func.__name__}")
    
    def add_validation_rule(self, validation_func: Callable[[DataRecord], bool]):
        """Add validation rule for data quality"""
        self.validation_rules.append(validation_func)
        print(f"✅ Added validation rule: {validation_func.__name__}")
    
    @intelligent_cache(max_size=50)
    def _apply_transformations(self, record: DataRecord) -> DataRecord:
        """Apply all registered transformations with caching"""
        
        current_record = record
        
        # Apply all transformation functions
        for transform_func in self.transformation_functions:
            current_record = transform_func(current_record)
        
        return current_record
    
    def _validate_record(self, record: DataRecord) -> bool:
        """Validate record against all registered rules"""
        
        for validation_func in self.validation_rules:
            if not validation_func(record):
                return False
        
        return True
    
    def process_batch(self, batch: List[DataRecord]) -> List[DataRecord]:
        """Process batch with transformations and validation"""
        
        processed_records = []
        
        for record in batch:
            # Apply transformations
            transformed_record = self._apply_transformations(record)
            
            # Validate record
            if self._validate_record(transformed_record):
                processed_records.append(transformed_record)
            else:
                print(f"⚠️  Record {record.id} failed validation")
        
        return processed_records

# ================================
# 6. FUNCTIONAL PROGRAMMING UTILITIES
# ================================

def compose_transformations(*transforms: Callable) -> Callable:
    """Compose multiple transformation functions into a pipeline"""
    def composed_transform(record: DataRecord) -> DataRecord:
        result = record
        for transform in transforms:
            result = transform(result)
        return result
    
    return composed_transform

def create_filter_predicate(**conditions) -> Callable[[DataRecord], bool]:
    """Create a filter predicate based on conditions"""
    def predicate(record: DataRecord) -> bool:
        for key, expected_value in conditions.items():
            if key in record.features:
                if record.features[key] != expected_value:
                    return False
        return True
    
    return predicate

# Transformation functions using functional programming
def normalize_numerical_features(record: DataRecord) -> DataRecord:
    """Normalize numerical features in the record"""
    normalized_features = record.features.copy()
    
    for key, value in record.features.items():
        if isinstance(value, (int, float)) and key != 'id':
            # Simple min-max normalization (in practice, would use dataset statistics)
            normalized_features[f"{key}_normalized"] = max(0, min(1, value / 100))
    
    return DataRecord(
        id=record.id,
        features=normalized_features,
        target=record.target,
        metadata={**record.metadata, 'normalized': True}
    )

def add_derived_features(record: DataRecord) -> DataRecord:
    """Add derived features based on existing ones"""
    enhanced_features = record.features.copy()
    
    # Create interaction features
    if 'age' in record.features and 'income' in record.features:
        enhanced_features['age_income_ratio'] = record.features['age'] / max(record.features['income'] / 1000, 1)
    
    if 'experience' in record.features:
        enhanced_features['is_experienced'] = 1 if record.features['experience'] > 3 else 0
    
    return DataRecord(
        id=record.id,
        features=enhanced_features,
        target=record.target,
        metadata={**record.metadata, 'derived_features_added': True}
    )

# ================================
# 7. DEMONSTRATION AND TESTING
# ================================

def generate_sample_data(num_records: int = 1000) -> List[Dict]:
    """Generate sample data for testing the pipeline"""
    
    import random
    random.seed(42)
    
    sample_data = []
    
    for i in range(num_records):
        record = {
            'id': i,
            'features': {
                'age': random.randint(18, 65),
                'income': random.randint(25000, 150000),
                'experience': random.randint(0, 20),
                'education_level': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'location': random.choice(['Urban', 'Suburban', 'Rural'])
            },
            'target': random.random()  # Random target for supervised learning
        }
        sample_data.append(record)
    
    return sample_data

def main():
    """Main function demonstrating the advanced ML data pipeline"""
    
    print("🎯 DEMONSTRATING ADVANCED ML DATA PIPELINE")
    print("-" * 50)
    
    # Generate sample data
    sample_data = generate_sample_data(500)
    print(f"📊 Generated {len(sample_data)} sample records")
    
    # Create processor
    processor = SmartMLDataProcessor()
    
    # Add custom transformations using functional programming
    composed_transform = compose_transformations(
        normalize_numerical_features,
        add_derived_features
    )
    
    processor.add_transformation(composed_transform)
    
    # Add validation rules
    def validate_age_range(record: DataRecord) -> bool:
        return 18 <= record.features.get('age', 0) <= 65
    
    def validate_income_positive(record: DataRecord) -> bool:
        return record.features.get('income', 0) > 0
    
    processor.add_validation_rule(validate_age_range)
    processor.add_validation_rule(validate_income_positive)
    
    # Configure pipeline
    original_config = processor.config.copy()
    
    # Test with different configurations
    with temporary_config_override(processor.config, batch_size=50, error_tolerance=0.1):
        
        # Process the dataset
        processed_records = processor.process_dataset(sample_data)
        
        print(f"\n📈 Processing Results:")
        print(f"   Original records: {len(sample_data)}")
        print(f"   Processed records: {len(processed_records)}")
        print(f"   Processing efficiency: {len(processed_records)/len(sample_data)*100:.1f}%")
        
        # Show sample processed record
        if processed_records:
            sample_record = processed_records[0]
            print(f"\n📋 Sample Processed Record:")
            print(f"   ID: {sample_record.id}")
            print(f"   Features: {len(sample_record.features)} total")
            print(f"   Sample features: {list(sample_record.features.keys())[:5]}...")
            print(f"   Metadata: {sample_record.metadata}")
        
        # Get processing summary
        summary = processor.get_processing_summary()
        print(f"\n📊 Processing Summary:")
        print(f"   Processor: {summary['processor_name']}")
        print(f"   Total runs: {summary['total_runs']}")
        print(f"   Latest stats: {summary['latest_stats']}")
        
        # Test caching performance
        print(f"\n💾 Cache Performance Test:")
        
        # Process same data again to test caching
        test_record = processed_records[0]
        
        # First call - should compute
        result1 = processor._apply_transformations(test_record)
        
        # Second call - should use cache
        result2 = processor._apply_transformations(test_record)
        
        cache_info = processor._apply_transformations.cache_info()
        print(f"   Cache info: {cache_info}")
    
    print(f"\n✅ Advanced ML Data Pipeline demonstration completed!")
    
    return processor, processed_records

# ================================
# 8. PERFORMANCE BENCHMARKING
# ================================

def benchmark_pipeline_performance():
    """Benchmark different pipeline configurations"""
    
    print("\n🏃‍♂️ PIPELINE PERFORMANCE BENCHMARKING")
    print("-" * 40)
    
    sample_data = generate_sample_data(1000)
    
    # Test different batch sizes
    batch_sizes = [50, 100, 200]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔬 Testing batch size: {batch_size}")
        
        processor = SmartMLDataProcessor()
        processor.config['batch_size'] = batch_size
        
        # Simple transformation for consistent testing
        processor.add_transformation(normalize_numerical_features)
        processor.add_validation_rule(lambda r: r.features.get('age', 0) > 0)
        
        start_time = time.time()
        processed_records = processor.process_dataset(sample_data)
        end_time = time.time()
        
        results[batch_size] = {
            'processing_time': end_time - start_time,
            'records_processed': len(processed_records),
            'throughput': len(processed_records) / (end_time - start_time)
        }
        
        print(f"   ⏱️  Processing time: {results[batch_size]['processing_time']:.3f}s")
        print(f"   📊 Throughput: {results[batch_size]['throughput']:.1f} records/sec")
    
    # Find optimal batch size
    optimal_batch_size = max(results.keys(), key=lambda k: results[k]['throughput'])
    print(f"\n🏆 Optimal batch size: {optimal_batch_size}")
    print(f"   Best throughput: {results[optimal_batch_size]['throughput']:.1f} records/sec")
    
    return results

# ================================
# RUN THE PIPELINE
# ================================

if __name__ == "__main__":
    # Run main demonstration
    processor, processed_data = main()
    
    # Run performance benchmarking
    benchmark_results = benchmark_pipeline_performance()
    
    print("\n" + "="*50)
    print("🏆 DAY 8 MINI PROJECT COMPLETED!")
    print("="*50)
    
    print("\n💪 Advanced Python Concepts Applied:")
    concepts_applied = [
        "✅ Decorators - Performance monitoring and intelligent caching",
        "✅ Generators - Memory-efficient data streaming and processing",
        "✅ Context Managers - Pipeline execution and configuration management", 
        "✅ Advanced OOP - Abstract base classes and extensible design",
        "✅ Functional Programming - Transformation composition and predicates",
        "✅ Type Hints - Professional code documentation and IDE support",
        "✅ Error Handling - Robust pipeline with configurable error tolerance",
        "✅ Performance Optimization - Caching, batching, and benchmarking"
    ]
    
    for concept in concepts_applied:
        print(f"  {concept}")
    
    print(f"\n🎯 Professional Capabilities Demonstrated:")
    capabilities = [
        "Memory-efficient processing of large datasets",
        "Configurable and extensible ML pipeline architecture", 
        "Advanced caching strategies for performance optimization",
        "Comprehensive error handling and monitoring",
        "Functional programming patterns for data transformations",
        "Professional code organization and documentation"
    ]
    
    for capability in capabilities:
        print(f"  🚀 {capability}")
    
    print(f"\n📊 Project Results:")
    if 'processed_data' in locals() and processed_data:
        print(f"  📈 Successfully processed {len(processed_data)} records")
        print(f"  ⚡ Pipeline configured for optimal performance")
        print(f"  💾 Intelligent caching implemented and tested")
        print(f"  🔧 Modular and extensible design pattern achieved")
    
    print(f"\n🌟 Ready for Day 9: Introduction to Machine Learning!")
    print(f"💪 Advanced Python skills will accelerate ML algorithm learning!")