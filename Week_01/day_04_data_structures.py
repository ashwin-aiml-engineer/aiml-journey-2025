# Day 4: Data Structures - Lists, Dictionaries, Sets, Tuples
print("=== Data Structures Practice ===")

# LISTS - Ordered, mutable collections
print("\n1. LISTS - Ordered and Mutable")
print("-" * 40)

# Creating lists
skills = ["Python", "Machine Learning", "Data Science"]
numbers = [1, 2, 3, 4, 5]
mixed_list = ["Ashwin", 25, True, 3.14, ["nested", "list"]]

print(f"Skills list: {skills}")
print(f"Numbers: {numbers}")
print(f"Mixed types: {mixed_list}")

# List operations
skills.append("Deep Learning")  # Add to end
skills.insert(1, "Statistics")  # Insert at index
skills.remove("Python")         # Remove by value
popped_skill = skills.pop()     # Remove and return last item

print(f"Modified skills: {skills}")
print(f"Popped skill: {popped_skill}")

# List indexing and slicing
programming_languages = ["Python", "Java", "C++", "JavaScript", "R", "Go"]
print(f"First language: {programming_languages[0]}")
print(f"Last language: {programming_languages[-1]}")
print(f"First 3: {programming_languages[:3]}")
print(f"Last 2: {programming_languages[-2:]}")
print(f"Every 2nd: {programming_languages[::2]}")

# List comprehensions - Powerful way to create lists
squares = [x**2 for x in range(10)]
even_numbers = [x for x in range(20) if x % 2 == 0]
uppercase_skills = [skill.upper() for skill in skills]

print(f"Squares: {squares}")
print(f"Even numbers: {even_numbers}")
print(f"Uppercase skills: {uppercase_skills}")

# Advanced list operations
learning_hours = [3, 2, 4, 3, 5, 2, 4, 6, 3, 4]
print(f"Total hours: {sum(learning_hours)}")
print(f"Average hours: {sum(learning_hours)/len(learning_hours):.1f}")
print(f"Max hours: {max(learning_hours)}")
print(f"Min hours: {min(learning_hours)}")

# DICTIONARIES - Key-value pairs, unordered, mutable
print("\n2. DICTIONARIES - Key-Value Pairs")
print("-" * 40)

# Creating dictionaries
student = {
    "name": "Ashwin Shetty",
    "age": 25,
    "background": "Mechanical Engineering",
    "skills": ["Python", "ML", "Data Science"],
    "current_day": 4
}

# Different ways to create dictionaries
grades = dict(python=85, statistics=90, ml=88)
course_hours = dict([("Python", 25), ("ML", 40), ("Statistics", 20)])

print(f"Student info: {student}")
print(f"Grades: {grades}")
print(f"Course hours: {course_hours}")

# Dictionary operations
student["email"] = "ashwin@email.com"  # Add new key
student["age"] = 26                    # Update existing key
student.update({"location": "Mumbai", "experience": 0})  # Add multiple

print(f"Updated student: {student}")

# Accessing dictionary values
print(f"Name: {student['name']}")
print(f"Skills: {student.get('skills', 'Not found')}")  # Safe access
print(f"Salary: {student.get('salary', 'Not specified')}")  # Default value

# Dictionary methods
print(f"Keys: {list(student.keys())}")
print(f"Values: {list(student.values())}")
print(f"Items: {list(student.items())}")

# Dictionary comprehensions
skill_ratings = {"Python": 7, "ML": 5, "Statistics": 6, "Deep Learning": 3}
high_skills = {skill: rating for skill, rating in skill_ratings.items() if rating >= 6}
skill_categories = {skill: "Beginner" if rating <= 3 else "Intermediate" if rating <= 6 else "Advanced" 
                   for skill, rating in skill_ratings.items()}

print(f"High skills: {high_skills}")
print(f"Skill categories: {skill_categories}")

# SETS - Unique elements, unordered, mutable
print("\n3. SETS - Unique Elements")
print("-" * 40)

# Creating sets
python_skills = {"variables", "functions", "classes", "modules"}
ml_skills = {"regression", "classification", "clustering", "neural networks"}
data_skills = {"pandas", "numpy", "matplotlib", "statistics"}

# Set from list (removes duplicates)
all_technologies = ["Python", "ML", "Statistics", "Python", "Deep Learning", "ML"]
unique_technologies = set(all_technologies)

print(f"Python skills: {python_skills}")
print(f"Unique technologies: {unique_technologies}")

# Set operations
python_skills.add("decorators")         # Add single element
python_skills.update(["generators", "context managers"])  # Add multiple
python_skills.discard("modules")        # Remove if exists (no error if not found)

print(f"Updated Python skills: {python_skills}")

# Mathematical set operations
beginner_topics = {"variables", "functions", "loops", "conditionals"}
intermediate_topics = {"classes", "modules", "decorators", "generators"}
advanced_topics = {"metaclasses", "asyncio", "decorators", "generators"}

# Union (all unique elements from both sets)
all_topics = beginner_topics | intermediate_topics | advanced_topics
print(f"All topics: {all_topics}")

# Intersection (common elements)
common_topics = intermediate_topics & advanced_topics
print(f"Common intermediate/advanced: {common_topics}")

# Difference (elements in first set but not second)
only_intermediate = intermediate_topics - advanced_topics
print(f"Only intermediate: {only_intermediate}")

# TUPLES - Ordered, immutable collections
print("\n4. TUPLES - Ordered and Immutable")
print("-" * 40)

# Creating tuples
coordinates = (10, 20)
student_info = ("Ashwin", 25, "Mechanical Engineering")
single_tuple = (42,)  # Note the comma for single element tuple

print(f"Coordinates: {coordinates}")
print(f"Student info: {student_info}")
print(f"Single tuple: {single_tuple}")

# Tuple unpacking
name, age, background = student_info
x, y = coordinates

print(f"Unpacked: name={name}, age={age}, background={background}")
print(f"Coordinates: x={x}, y={y}")

# Named tuples for better readability
from collections import namedtuple

Student = namedtuple('Student', ['name', 'age', 'background', 'skills'])
ashwin = Student("Ashwin Shetty", 25, "Mechanical Engineering", ["Python", "ML"])

print(f"Named tuple: {ashwin}")
print(f"Name: {ashwin.name}, Skills: {ashwin.skills}")

# Tuple methods
numbers_tuple = (1, 2, 3, 2, 4, 2, 5)
print(f"Count of 2: {numbers_tuple.count(2)}")
print(f"Index of 4: {numbers_tuple.index(4)}")

# PRACTICAL EXAMPLES - Combining Data Structures
print("\n5. PRACTICAL EXAMPLES - Real-world Usage")
print("-" * 40)

# Learning management system using multiple data structures
learning_system = {
    "students": {
        "STU001": {
            "name": "Ashwin Shetty",
            "completed_courses": ["Python Basics", "OOP", "File Handling"],
            "skills": {"Python": 7, "OOP": 6, "File Handling": 5},
            "certificates": {"Python Fundamentals", "OOP Mastery"}
        },
        "STU002": {
            "name": "Priya Sharma",
            "completed_courses": ["Python Basics", "Statistics"],
            "skills": {"Python": 8, "Statistics": 7},
            "certificates": {"Python Advanced"}
        }
    },
    "courses": {
        "Python Basics": {"duration": 20, "difficulty": "Beginner"},
        "OOP": {"duration": 15, "difficulty": "Intermediate"},
        "Statistics": {"duration": 25, "difficulty": "Intermediate"}
    }
}

# Analyze the learning system
def analyze_learning_system(system):
    total_students = len(system["students"])
    all_skills = set()
    all_certificates = set()
    
    for student_data in system["students"].values():
        all_skills.update(student_data["skills"].keys())
        all_certificates.update(student_data["certificates"])
    
    print(f"Total students: {total_students}")
    print(f"Unique skills taught: {all_skills}")
    print(f"Certificates available: {all_certificates}")
    
    # Find top performers
    student_scores = {}
    for student_id, data in system["students"].items():
        avg_skill_level = sum(data["skills"].values()) / len(data["skills"])
        student_scores[data["name"]] = avg_skill_level
    
    top_student = max(student_scores, key=student_scores.get)
    print(f"Top performer: {top_student} (avg skill: {student_scores[top_student]:.1f})")

analyze_learning_system(learning_system)

# Data structure performance comparison
print("\n6. PERFORMANCE EXAMPLES")
print("-" * 40)

import time

# List vs Set for membership testing
large_list = list(range(10000))
large_set = set(range(10000))
search_item = 9999

# Time list search
start_time = time.time()
result = search_item in large_list
list_time = time.time() - start_time

# Time set search  
start_time = time.time()
result = search_item in large_set
set_time = time.time() - start_time

print(f"List search time: {list_time:.6f} seconds")
print(f"Set search time: {set_time:.6f} seconds")
print(f"Set is {list_time/set_time:.1f}x faster for membership testing")

# When to use each data structure
usage_guide = {
    "Lists": [
        "When order matters",
        "Need to access by index", 
        "Allow duplicate values",
        "Need to modify frequently"
    ],
    "Dictionaries": [
        "Key-value relationships",
        "Fast lookups by key",
        "Represent structured data",
        "Cache or mapping data"
    ],
    "Sets": [
        "Unique elements only",
        "Fast membership testing",
        "Mathematical operations",
        "Remove duplicates"
    ],
    "Tuples": [
        "Immutable data",
        "Return multiple values",
        "Dictionary keys",
        "Coordinates or pairs"
    ]
}

print("\n7. USAGE GUIDELINES")
print("-" * 40)
for data_type, uses in usage_guide.items():
    print(f"{data_type}:")
    for use in uses:
        print(f"  - {use}")
    print()

# Advanced example: AI/ML data processing pipeline
print("8. AI/ML DATA PROCESSING PIPELINE")
print("-" * 40)

# Simulated dataset
raw_data = [
    {"name": "Alice", "skills": ["Python", "ML", "Stats"], "experience": 2},
    {"name": "Bob", "skills": ["Python", "Deep Learning"], "experience": 3},
    {"name": "Charlie", "skills": ["R", "Stats", "ML"], "experience": 1},
    {"name": "Diana", "skills": ["Python", "ML", "Deep Learning"], "experience": 4}
]

# Data processing pipeline using different data structures
def process_candidates(candidates):
    # Use set to find all unique skills
    all_skills = set()
    for candidate in candidates:
        all_skills.update(candidate["skills"])
    
    # Use dictionary to count skill frequency
    skill_count = {}
    for skill in all_skills:
        skill_count[skill] = sum(1 for candidate in candidates if skill in candidate["skills"])
    
    # Use list comprehension to filter experienced candidates
    experienced = [c for c in candidates if c["experience"] >= 3]
    
    # Use tuple for results (immutable summary)
    summary = (
        len(candidates),           # total candidates
        len(all_skills),          # unique skills
        len(experienced),         # experienced candidates
        max(skill_count, key=skill_count.get)  # most common skill
    )
    
    return {
        "all_skills": all_skills,
        "skill_frequency": skill_count,
        "experienced_candidates": experienced,
        "summary": summary
    }

results = process_candidates(raw_data)
total, unique_skills, experienced_count, top_skill = results["summary"]

print(f"Processed {total} candidates")
print(f"Found {unique_skills} unique skills")
print(f"Most common skill: {top_skill}")
print(f"Experienced candidates: {experienced_count}")
print(f"Skill frequency: {results['skill_frequency']}")

print("\nDay 4 Data Structures practice completed!")