# Basic function definition
def greet():
    """Simple greeting function"""
    print("Hello, AI/ML learner!")

def greet_person(name):
    """Function with parameter"""
    print(f"Hello, {name}! Ready for Day 2?")

# Call functions
greet()
greet_person("Ashwin")

# Function with return value
def calculate_completion_percentage(days_completed, total_days):
    """Calculate learning progress percentage"""
    percentage = (days_completed / total_days) * 100
    return percentage

# Function with multiple parameters and return
def evaluate_learning_progress(day, topics_covered, hours_studied):
    """Evaluate daily learning progress"""
    efficiency = topics_covered / hours_studied if hours_studied > 0 else 0
    status = "Excellent" if efficiency >= 1.5 else "Good" if efficiency >= 1.0 else "Needs improvement"
    
    return {
        'day': day,
        'efficiency': efficiency,
        'status': status,
        'recommendation': get_recommendation(efficiency)
    }

def get_recommendation(efficiency):
    """Get learning recommendation based on efficiency"""
    if efficiency >= 1.5:
        return "Keep up the excellent pace!"
    elif efficiency >= 1.0:
        return "Good progress, try to increase focus"
    else:
        return "Slow down and focus on understanding"

# Function with default parameters
def create_study_plan(subject, hours_per_day=2, days_per_week=5):
    """Create a study plan with default values"""
    weekly_hours = hours_per_day * days_per_week
    return f"Study plan for {subject}: {hours_per_day} hours/day, {days_per_week} days/week = {weekly_hours} hours/week"

# Functions for mathematical calculations
def factorial(n):
    """Calculate factorial recursively"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_list = [0, 1]
    for i in range(2, n):
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    return fib_list

def is_prime(num):
    """Check if a number is prime"""
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# AI/ML related utility functions
def calculate_model_accuracy(correct_predictions, total_predictions):
    """Calculate model accuracy percentage"""
    if total_predictions == 0:
        return 0
    return (correct_predictions / total_predictions) * 100

def estimate_learning_time(current_skill_level, target_level, hours_per_day):
    """Estimate time needed to reach target skill level"""
    skill_gap = target_level - current_skill_level
    days_needed = (skill_gap * 10) / hours_per_day  # Rough estimation
    return max(1, int(days_needed))

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
add = lambda x, y: x + y
is_even = lambda x: x % 2 == 0

# Testing all functions
print("=== Function Testing ===")

# Test basic functions
progress = calculate_completion_percentage(2, 90)
print(f"Current progress: {progress:.1f}%")

# Test complex function
learning_data = evaluate_learning_progress(2, 3, 3)
print(f"Day {learning_data['day']} evaluation:")
print(f"Efficiency: {learning_data['efficiency']:.2f}")
print(f"Status: {learning_data['status']}")
print(f"Recommendation: {learning_data['recommendation']}")

# Test default parameters
plan1 = create_study_plan("Python")
plan2 = create_study_plan("Machine Learning", 3, 6)
print(f"n{plan1}")
print(f"{plan2}")

# Test mathematical functions
print(f"nFactorial of 5: {factorial(5)}")
print(f"First 8 Fibonacci numbers: {fibonacci(8)}")
print(f"Is 17 prime? {is_prime(17)}")
print(f"Is 25 prime? {is_prime(25)}")

# Test AI/ML functions
accuracy = calculate_model_accuracy(85, 100)
print(f"Model accuracy: {accuracy}%")

learning_time = estimate_learning_time(2, 8, 3)
print(f"Estimated learning time: {learning_time} days")

# Test lambda functions
print(f"nSquare of 7: {square(7)}")
print(f"5 + 3 = {add(5, 3)}")
print(f"Is 8 even? {is_even(8)}")

# Function with variable arguments
def log_learning_session(*topics, **details):
    """Log a learning session with variable arguments"""
    print(f"Learning session on Day {details.get('day', 'Unknown')}")
    print(f"Duration: {details.get('duration', 'Not specified')} hours")
    print("Topics covered:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    if details.get('notes'):
        print(f"Notes: {details['notes']}")

# Test variable arguments function
log_learning_session("Control Structures", "Loops", "Functions", 
                     day=2, duration=3, notes="Good progress on fundamentals")