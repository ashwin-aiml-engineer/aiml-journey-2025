# Basic for loop with range
print(f"Counting from 1 to 1000:")
for i in range(1,1001,):
    print(f"Count: {i}")
print("Counting from 1 to 1000 completed.")

# For loop with step
print(f"Counting from 1 to 1000:")
for i in range(1,1001,2):
    print(f"Count: {i}")
print("Counting from 1 to 1000 completed.")

# For loop with negative step
print(f"Counting down from 1000 to 1:")
for i in range(1000,0,-1):
    print(f"Count: {i}")
print("Counting down from 1000 to 1 completed.")

# For loop with strings
name = "AIML"
print(f"\\nLetters in {name}:")
for letter in name:
    print(f"Letter: {letter}")

# For loop with lists
skills = ["Python", "Mathematics", "Statistics", "Machine Learning"]
print("\\nSkills I'm learning:")
for i, skill in enumerate(skills, 1):
    print(f"{i}. {skill}")

# While loop - basic counting
print("\\nCountdown using while loop:")
count = 5
while count > 0:
    print(f"T-minus {count}")
    count -= 1
print("Launch!")

# While loop - user input validation
print("\\nPassword strength checker:")
while True:
    password = input("Enter a strong password (min 8 characters): ")
    if len(password) >= 8:
        print("Password accepted!")
        break
    else:
        print("Password too short. Try again.")

# Nested loops - multiplication table
print("\\nMultiplication table (1-5):")
for i in range(1, 6):
    for j in range(1, 6):
        result = i * j
        print(f"{result:2d}",end=" ")
    print() 

# Nested loops - pattern printing case-1
print("\\nPattern printing:")
print("Right triangle:")
for i in range(1, 6):
    print("* " * i)

# Nested loops - pattern printing case-2
print("\\nNumber pyramid:")
for i in range(1, 6):
    print(" " * (5-i) + " ".join(str(x) for x in range(1, i+1)))

# Loop with break and continue
print("\\nFinding first even number in a list:")
numbers = [1, 3, 5, 7, 8, 9]
for num in numbers:
    if num % 2 != 0:
        continue
    print(f"First even number found: {num}")
    break

# Practical example - calculating learning progress
total_days = 90
current_day = 2
print(f"\\nDaily Progress Tracker:")
for day in range(1, current_day + 1):
    percentage = (day / total_days) * 100
    print(f"Day {day}: {percentage:.1f}% complete")