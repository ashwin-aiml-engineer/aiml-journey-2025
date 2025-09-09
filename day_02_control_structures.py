# Case 1 : Basic if-else
age = int(input("Enter your age :"))
if age >= 18:
    print("You are eligible to vote.")
else:
    print("You are not eligible to vote.")

# Case 2 : Multiple Conditions with elif
marks = float(input("Enter your marks(0-100) : "))
if marks >= 90:
    grade = "A+"
    message = "Outstanding Performance"
elif marks >= 80:
    grade = "A"
    message = "Excellent"
elif marks >= 70:
    grade = "B"
    message = "Good"
elif marks >= 60:
    grade = "C"
    message = "Average"
elif marks >= 50:
    grade = "D"
    message = "Poor"
else :
    grade = "F"
    message = "Failed."
print(f"Grade: {grade}")
print(f"Feedback: {message}") 

# Case 3 : Nested Conditions
username = input("Enter your username : ")
password = input("Enter your password : ")  
if username == "Ashwin":
    if password == "12345678":
        print("Login Successful")
    else:
        print("Incorrect Password")
else:
    print("Username not found")

# Case 4 : Logical Operators in conditions
hour = int(input("Enter the current hour (0-23)) : "))
day = input( "Enter the day of the week : ")
if (hour <9 or hour >17) and (day == "Saturday" or day == "Sunday" or day == "saturday"):
    print("The office is closed.")
else:
    print("The office is open.")

# Case 5 : Ternary Operator
temperature = float(input("Enter the temperature in Celsius : "))
weather = "Extreme Hot" if temperature >= 60 else "Hot" if temperature >= 30 else "Warm" if temperature >= 20 else "Cool" if temperature >= 10 else "Cold"
print(f"The weather is {weather}.")

# Case 6 : Match-Case 
day = input("Enter the day of the week : ").lower()
match day:
    case "monday":
        print("Start of the work week.")
    case "wednesday":
        print("Midweek day.")
    case "friday":
        print("End of the work week.")
    case "saturday" | "sunday":
        print("It's the weekend!")
    case _:
        print("Just another weekday.")

# Case 7 : Complex condition example - AI/ML career readiness check
python_score = int(input("Rate your Python knowledge (1-10): "))
math_score = int(input("Rate your math skills (1-10): "))
dedication = input("Are you dedicated to learning? (yes/no): ").lower()
if python_score >= 7 and math_score >= 6 and dedication == "yes":
    print("You're ready for advanced AI/ML topics!")
elif python_score >= 5 and math_score >= 4:
    print("Good foundation, keep practicing fundamentals")
elif dedication == "yes":
    print("Great attitude! Focus on building technical skills")
else:
    print("Start with basics and build consistency")