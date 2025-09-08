# Input/Output Practice
print("=== Personal Information Collector ===")

# Get user input
user_name = input("Enter your name: ")
user_age = input("Enter your age: ")
user_city = input("Enter your city: ")
user_goal = input("What do you want to become? ")

# Convert age to integer for calculations
age_num = int(user_age)
retirement_age = 60
years_to_retirement = retirement_age - age_num

# Display formatted output
print(f"\n=== Summary ===")
print(f"Hello {user_name}!")
print(f"You are {user_age} years old")
print(f"You live in {user_city}")
print(f"Your goal is to become a {user_goal}")
print(f"You have {years_to_retirement} years until retirement")

# Practice with different input types
favorite_number = float(input("Enter your favorite number: "))
print(f"Your favorite number squared is: {favorite_number ** 2}")
