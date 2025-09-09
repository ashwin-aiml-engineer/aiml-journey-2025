# Problem 1: Student Grade Management System
def calculate_grade(marks):
    """Calculate letter grade from marks"""
    if marks >= 90:
        return "A+", "Outstanding"
    elif marks >= 80:
        return "A", "Excellent"
    elif marks >= 70:
        return "B", "Good"
    elif marks >= 60:
        return "C", "Average"
    elif marks >= 50:
        return "D", "Below Average"
    else:
        return "F", "Fail"

def student_grade_system():
    """Complete student grade management system"""
    print("=== Student Grade Management System ===")
    students = {}
    
    while True:
        print("1. Add student")
        print("2. View all students")
        print("3. Calculate class average")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            name = input("Enter student name: ")
            try:
                marks = float(input("Enter marks (0-100): "))
                if 0 <= marks <= 100:
                    grade, description = calculate_grade(marks)
                    students[name] = {"marks": marks, "grade": grade, "description": description}
                    print(f"Added {name} - Marks: {marks}, Grade: {grade}")
                else:
                    print("Marks should be between 0-100")
            except ValueError:
                print("Please enter valid marks")
                
        elif choice == "2":
            if not students:
                print("No students added yet")
            else:
                print("\\n--- Student Records ---")
                for name, data in students.items():
                    print(f"{name}: {data['marks']} marks, Grade {data['grade']} ({data['description']})")
                    
        elif choice == "3":
            if not students:
                print("No students to calculate average")
            else:
                total_marks = sum(data['marks'] for data in students.values())
                average = total_marks / len(students)
                grade, description = calculate_grade(average)
                print(f"Class average: {average:.2f} marks, Grade {grade}")
                
        elif choice == "4":
            print("Exiting student management system")
            break
        else:
            print("Invalid choice")

# Problem 2: Number Pattern Generator
def generate_patterns():
    """Generate various number patterns"""
    print("\\n=== Pattern Generator ===")
    
    size = int(input("Enter pattern size: "))
    
    print(f"\\nPattern 1 - Right Triangle:")
    for i in range(1, size + 1):
        for j in range(1, i + 1):
            print(j, end=" ")
        print()
    
    print(f"\\nPattern 2 - Number Pyramid:")
    for i in range(1, size + 1):
        # Print spaces
        for j in range(size - i):
            print(" ", end=" ")
        # Print numbers
        for j in range(1, i + 1):
            print(j, end=" ")
        print()
    
    print(f"\\nPattern 3 - Diamond:")
    # Upper half
    for i in range(1, size + 1):
        for j in range(size - i):
            print(" ", end="")
        for j in range(1, i + 1):
            print("*", end=" ")
        print()
    # Lower half
    for i in range(size - 1, 0, -1):
        for j in range(size - i):
            print(" ", end="")
        for j in range(1, i + 1):
            print("*", end=" ")
        print()

# Problem 3: Learning Progress Tracker
def learning_tracker():
    """Track learning progress over multiple days"""
    print("\\n=== Learning Progress Tracker ===")
    
    total_days = 90
    progress_data = {}
    
    while True:
        print("\\n1. Log today's progress")
        print("2. View progress summary")
        print("3. Calculate remaining time")
        print("4. Exit tracker")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            day = int(input("Enter day number (1-90): "))
            if 1 <= day <= 90:
                topics = int(input("Topics covered today: "))
                hours = float(input("Hours studied: "))
                confidence = int(input("Confidence level (1-10): "))
                
                progress_data[day] = {
                    'topics': topics,
                    'hours': hours,
                    'confidence': confidence,
                    'efficiency': topics / hours if hours > 0 else 0
                }
                print(f"Day {day} logged successfully!")
            else:
                print("Day should be between 1-90")
                
        elif choice == "2":
            if not progress_data:
                print("No progress data available")
            else:
                print("\\n--- Progress Summary ---")
                total_topics = sum(data['topics'] for data in progress_data.values())
                total_hours = sum(data['hours'] for data in progress_data.values())
                avg_confidence = sum(data['confidence'] for data in progress_data.values()) / len(progress_data)
                
                print(f"Days logged: {len(progress_data)}")
                print(f"Total topics covered: {total_topics}")
                print(f"Total hours studied: {total_hours:.1f}")
                print(f"Average confidence: {avg_confidence:.1f}/10")
                print(f"Overall efficiency: {total_topics/total_hours:.2f} topics/hour")
                
        elif choice == "3":
            current_day = max(progress_data.keys()) if progress_data else 1
            remaining_days = total_days - current_day
            completion_percentage = (current_day / total_days) * 100
            
            print(f"Current day: {current_day}")
            print(f"Days remaining: {remaining_days}")
            print(f"Progress: {completion_percentage:.1f}%")
            
            if completion_percentage < 25:
                print("Status: Just getting started!")
            elif completion_percentage < 50:
                print("Status: Building momentum!")
            elif completion_percentage < 75:
                print("Status: Halfway there!")
            else:
                print("Status: Almost finished!")
                
        elif choice == "4":
            break

# Problem 4: Simple Calculator with Memory
from math import factorial
def advanced_calculator():
    """Calculator with memory functions"""
    print("\\n=== Advanced Calculator ===")
    
    memory = 0
    
    while True:
        print(f"\\nMemory: {memory}")
        print("1. Basic calculation")
        print("2. Memory operations")
        print("3. Scientific functions")
        print("4. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            try:
                num1 = float(input("Enter first number: "))
                operation = input("Enter operation (+, -, *, /, **, %): ")
                num2 = float(input("Enter second number: "))
                
                if operation == "+":
                    result = num1 + num2
                elif operation == "-":
                    result = num1 - num2
                elif operation == "*":
                    result = num1 * num2
                elif operation == "/":
                    result = num1 / num2 if num2 != 0 else "Error: Division by zero"
                elif operation == "**":
                    result = num1 ** num2
                elif operation == "%":
                    result = num1 % num2 if num2 != 0 else "Error: Division by zero"
                else:
                    result = "Invalid operation"
                
                print(f"Result: {result}")
                if isinstance(result, (int, float)):
                    save_to_memory = input("Save to memory? (y/n): ")
                    if save_to_memory.lower() == 'y':
                        memory = result
                        
            except ValueError:
                print("Invalid input")
                
        elif choice == "2":
            print("Memory operations:")
            print("1. Store value")
            print("2. Recall memory")
            print("3. Clear memory")
            print("4. Add to memory")
            
            mem_choice = input("Enter choice: ")
            
            if mem_choice == "1":
                memory = float(input("Enter value to store: "))
            elif mem_choice == "2":
                print(f"Memory value: {memory}")
            elif mem_choice == "3":
                memory = 0
                print("Memory cleared")
            elif mem_choice == "4":
                value = float(input("Enter value to add: "))
                memory += value
                print(f"New memory value: {memory}")
                
        elif choice == "3":
            print("Scientific functions:")
            num = float(input("Enter number: "))
            
            print(f"Square: {num ** 2}")
            print(f"Square root: {num ** 0.5}")
            print(f"Cube: {num ** 3}")
            print(f"Factorial: {factorial(int(num)) if num >= 0 and num == int(num) else 'N/A'}")
            
        elif choice == "4":
            break

# Run all problems
if __name__ == "__main__":
    print("Choose a problem to solve:")
    print("1. Student Grade Management System")
    print("2. Pattern Generator")
    print("3. Learning Progress Tracker")
    print("4. Advanced Calculator")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        student_grade_system()
    elif choice == "2":
        generate_patterns()
    elif choice == "3":
        learning_tracker()
    elif choice == "4":
        advanced_calculator()
    else:
        print("Running all demos...")
        student_grade_system()
        generate_patterns()
        learning_tracker()
        advanced_calculator()