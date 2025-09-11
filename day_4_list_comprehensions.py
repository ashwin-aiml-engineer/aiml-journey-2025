# Day 4: Advanced List Operations & Comprehensions
print("=== Advanced List Operations & Comprehensions ===")

# BASIC LIST COMPREHENSIONS
print("\n1. BASIC LIST COMPREHENSIONS")
print("-" * 40)

# Traditional way vs List comprehension
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Traditional way
squares_traditional = []
for x in numbers:
    squares_traditional.append(x**2)

# List comprehension way
squares_comprehension = [x**2 for x in numbers]

print(f"Traditional squares: {squares_traditional}")
print(f"Comprehension squares: {squares_comprehension}")

# More examples
even_squares = [x**2 for x in numbers if x % 2 == 0]
odd_cubes = [x**3 for x in numbers if x % 2 == 1]
string_lengths = [len(word) for word in ["Python", "Machine", "Learning", "AI"]]

print(f"Even squares: {even_squares}")
print(f"Odd cubes: {odd_cubes}")
print(f"String lengths: {string_lengths}")

# CONDITIONAL LIST COMPREHENSIONS
print("\n2. CONDITIONAL LIST COMPREHENSIONS")
print("-" * 40)

# Filtering with conditions
grades = [85, 92, 78, 96, 85, 88, 79, 94, 87, 91]
passing_grades = [grade for grade in grades if grade >= 80]
grade_categories = ["Pass" if grade >= 80 else "Fail" for grade in grades]
letter_grades = ["A" if grade >= 90 else "B" if grade >= 80 else "C" if grade >= 70 else "F" 
                for grade in grades]

print(f"Original grades: {grades}")
print(f"Passing grades: {passing_grades}")
print(f"Pass/Fail: {grade_categories}")
print(f"Letter grades: {letter_grades}")

# AI/ML student performance analysis
student_scores = {
    "Ashwin": {"python": 85, "ml": 78, "stats": 92},
    "Priya": {"python": 92, "ml": 88, "stats": 85},
    "Rahul": {"python": 76, "ml": 82, "stats": 79},
    "Anita": {"python": 94, "ml": 91, "stats": 88}
}

# Find students with high average scores
high_performers = [name for name, scores in student_scores.items() 
                  if sum(scores.values()) / len(scores) >= 85]

# Get all Python scores above 80
good_python_students = [name for name, scores in student_scores.items() 
                       if scores["python"] >= 80]

print(f"High performers (avg >= 85): {high_performers}")
print(f"Good Python students (>= 80): {good_python_students}")

# NESTED LIST COMPREHENSIONS
print("\n3. NESTED LIST COMPREHENSIONS")
print("-" * 40)

# 2D matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Flatten matrix
flattened = [num for row in matrix for num in row]
print(f"Original matrix: {matrix}")
print(f"Flattened: {flattened}")

# Create multiplication table
multiplication_table = [[i * j for j in range(1, 6)] for i in range(1, 6)]
print("Multiplication table (5x5):")
for row in multiplication_table:
    print(row)

# Filter and transform nested structure
student_courses = [
    ["Python", "Statistics", "Linear Algebra"],
    ["Machine Learning", "Deep Learning"],
    ["Data Visualization", "SQL", "Python"]
]

# Get all unique courses
all_courses = list(set([course for student in student_courses for course in student]))
print(f"All unique courses: {all_courses}")

# Find courses with "Python" in the name
python_courses = [course for student in student_courses for course in student 
                 if "Python" in course]
print(f"Python-related courses: {python_courses}")

# ADVANCED TRANSFORMATIONS
print("\n4. ADVANCED TRANSFORMATIONS")
print("-" * 40)

# Text processing with list comprehensions
sentences = [
    "Machine learning is fascinating",
    "Python programming is powerful",
    "Data science requires statistics",
    "AI will transform industries"
]

# Word analysis
word_counts = [len(sentence.split()) for sentence in sentences]
uppercase_sentences = [sentence.upper() for sentence in sentences]
sentences_with_python = [sentence for sentence in sentences if "Python" in sentence]

print(f"Word counts: {word_counts}")
print(f"Python sentences: {sentences_with_python}")

# Extract specific information
words_starting_with_m = [word.lower() for sentence in sentences 
                        for word in sentence.split() 
                        if word.lower().startswith('m')]
print(f"Words starting with 'm': {words_starting_with_m}")

# Data cleaning example
messy_data = ["  Python  ", "ML", "", "  DATA SCIENCE  ", None, "ai", "  "]
cleaned_data = [item.strip().title() for item in messy_data 
               if item and item.strip()]
print(f"Messy data: {messy_data}")
print(f"Cleaned data: {cleaned_data}")

# DICTIONARY AND SET COMPREHENSIONS
print("\n5. DICTIONARY AND SET COMPREHENSIONS")
print("-" * 40)

# Dictionary comprehensions
skills = ["Python", "Machine Learning", "Statistics", "Deep Learning"]
skill_ratings = {skill: len(skill) % 5 + 5 for skill in skills}  # Mock ratings
advanced_skills = {skill: rating for skill, rating in skill_ratings.items() 
                  if rating >= 7}

print(f"Skill ratings: {skill_ratings}")
print(f"Advanced skills: {advanced_skills}")

# Create grade mappings
students = ["Alice", "Bob", "Charlie", "Diana"]
random_grades = [85, 92, 78, 88]
student_grades = {student: grade for student, grade in zip(students, random_grades)}
grade_letters = {student: "A" if grade >= 90 else "B" if grade >= 80 else "C"
                for student, grade in student_grades.items()}

print(f"Student grades: {student_grades}")
print(f"Letter grades: {grade_letters}")

# Set comprehensions
numbers_text = "The dataset contains 123 samples with 456 features and 789 labels"
digits = {int(char) for char in numbers_text if char.isdigit()}
unique_words = {word.lower() for word in numbers_text.split() 
               if word.isalpha() and len(word) > 3}

print(f"Unique digits: {digits}")
print(f"Unique long words: {unique_words}")

# PERFORMANCE OPTIMIZATION
print("\n6. PERFORMANCE OPTIMIZATION")
print("-" * 40)

import time

# Compare performance: traditional loop vs comprehension
def traditional_processing(data):
    result = []
    for x in data:
        if x % 2 == 0:
            result.append(x**2)
    return result

def comprehension_processing(data):
    return [x**2 for x in data if x % 2 == 0]

# Test with large dataset
large_data = list(range(100000))

# Time traditional approach
start_time = time.time()
result1 = traditional_processing(large_data)
traditional_time = time.time() - start_time

# Time comprehension approach
start_time = time.time()
result2 = comprehension_processing(large_data)
comprehension_time = time.time() - start_time

print(f"Traditional time: {traditional_time:.4f} seconds")
print(f"Comprehension time: {comprehension_time:.4f} seconds")
print(f"Comprehension is {traditional_time/comprehension_time:.2f}x faster")

# GENERATOR EXPRESSIONS (Memory Efficient)
print("\n7. GENERATOR EXPRESSIONS")
print("-" * 40)

# List comprehension vs Generator expression
list_comp = [x**2 for x in range(1000)]  # Creates list in memory
gen_exp = (x**2 for x in range(1000))    # Creates generator object

print(f"List comprehension type: {type(list_comp)}")
print(f"Generator expression type: {type(gen_exp)}")

# Memory usage comparison
import sys
list_size = sys.getsizeof(list_comp)
gen_size = sys.getsizeof(gen_exp)

print(f"List memory usage: {list_size} bytes")
print(f"Generator memory usage: {gen_size} bytes")
print(f"List uses {list_size/gen_size:.1f}x more memory")

# Using generators for large datasets
def process_large_dataset():
    # Generator for processing large amounts of data
    data_generator = (x**2 for x in range(1000000) if x % 1000 == 0)
    
    # Process in chunks
    total = 0
    count = 0
    for value in data_generator:
        total += value
        count += 1
        if count >= 10:  # Process first 10 values
            break
    
    return total, count

total, processed = process_large_dataset()
print(f"Processed {processed} values, total: {total}")

# REAL-WORLD ML DATA PROCESSING
print("\n8. REAL-WORLD ML DATA PROCESSING")
print("-" * 40)

# Simulate ML dataset preprocessing
raw_dataset = [
    {"age": 25, "salary": 50000, "experience": 2, "department": "Engineering"},
    {"age": 35, "salary": 75000, "experience": 8, "department": "Data Science"},
    {"age": 28, "salary": 60000, "experience": 4, "department": "Engineering"},
    {"age": 42, "salary": 95000, "experience": 15, "department": "Management"},
    {"age": 30, "salary": 70000, "experience": 6, "department": "Data Science"}
]

# Feature extraction using comprehensions
ages = [person["age"] for person in raw_dataset]
high_earners = [person for person in raw_dataset if person["salary"] > 65000]
experience_salary_ratio = [person["salary"] / person["experience"] 
                          for person in raw_dataset if person["experience"] > 0]

# Data transformation
normalized_ages = [(age - min(ages)) / (max(ages) - min(ages)) for age in ages]
department_counts = {}
for person in raw_dataset:
    dept = person["department"]
    department_counts[dept] = department_counts.get(dept, 0) + 1

print(f"Age range: {min(ages)} - {max(ages)}")
print(f"High earners (>65k): {len(high_earners)}")
print(f"Avg salary/experience ratio: {sum(experience_salary_ratio)/len(experience_salary_ratio):.0f}")
print(f"Department distribution: {department_counts}")

# Feature engineering
engineered_features = [
    {
        "age_group": "Young" if person["age"] < 30 else "Mid" if person["age"] < 40 else "Senior",
        "salary_tier": "High" if person["salary"] > 70000 else "Medium" if person["salary"] > 55000 else "Low",
        "experience_level": "Junior" if person["experience"] < 5 else "Senior",
        "is_tech": person["department"] in ["Engineering", "Data Science"]
    }
    for person in raw_dataset
]

print("Engineered features sample:")
for i, features in enumerate(engineered_features[:3]):
    print(f"Person {i+1}: {features}")

# COMMON PATTERNS AND BEST PRACTICES
print("\n9. COMMON PATTERNS AND BEST PRACTICES")
print("-" * 40)

# Pattern 1: Filtering and transforming
def filter_and_transform(data, filter_func, transform_func):
    return [transform_func(item) for item in data if filter_func(item)]

numbers = range(1, 21)
even_squares = filter_and_transform(numbers, lambda x: x % 2 == 0, lambda x: x**2)
print(f"Even squares: {even_squares}")

# Pattern 2: Grouping data
def group_by_key(data, key_func):
    groups = {}
    for item in data:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups

# Group students by performance level
performance_groups = group_by_key(
    raw_dataset, 
    lambda person: "High" if person["salary"] > 70000 else "Low"
)

print("Performance groups:")
for level, people in performance_groups.items():
    names = [p["department"] for p in people]
    print(f"{level}: {names}")

# Pattern 3: Data validation
def validate_data(dataset, validators):
    return [
        {**item, "valid": all(validator(item) for validator in validators)}
        for item in dataset
    ]

validators = [
    lambda x: x["age"] > 18,
    lambda x: x["salary"] > 0,
    lambda x: x["experience"] >= 0
]

validated_data = validate_data(raw_dataset, validators)
valid_count = sum(1 for item in validated_data if item["valid"])
print(f"Valid records: {valid_count}/{len(validated_data)}")

print("\nDay 4 Advanced List Operations completed!")
