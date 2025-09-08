#Problem 1: Simple Calculator
print("=== Problem 1: Simple Calculator ===")
num1 = float(input("Enter first number: "))
operation = input("Enter operation (+, -, *, /): ")
num2 = float(input("Enter second number: "))

if operation == "+":
    result = num1 + num2
elif operation == "-":
    result = num1 - num2
elif operation == "*":
    result = num1 * num2
elif operation == "/":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Error: Division by zero"
else:
    result = "Error: Invalid operation"

print(f"Result: {result}")

#Problem 2: Grade Calculator
print("\n=== Problem 2: Grade Calculator ===")
marks = float(input("Enter your marks (0-100): "))

if marks >= 90:
    grade = "A+"
elif marks >= 80:
    grade = "A"
elif marks >= 70:
    grade = "B"
elif marks >= 60:
    grade = "C"
elif marks >= 50:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")

#Problem 3: Learning Progress Tracker
print("\n=== Problem 3: Learning Progress Tracker ===")
total_days = 90
completed_days = int(input("How many days have you completed? "))
percentage = (completed_days / total_days) * 100

print(f"Progress: {completed_days}/{total_days} days")
print(f"Completion: {percentage:.1f}%")
print(f"Days remaining: {total_days - completed_days}")

if percentage < 25:
    message = "Just getting started! Keep going!"
elif percentage < 50:
    message = "Good progress! You're building momentum!"
elif percentage < 75:
    message = "Great work! More than halfway there!"
else:
    message = "Outstanding! You're almost there!"

print(f"Motivation: {message}")