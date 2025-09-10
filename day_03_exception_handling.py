print("=== Exception Handling and Advanced Concepts ===")

# Basic exception handling
def basic_exception_demo():
    """Demonstrate basic try-except blocks"""
    print("=== Basic Exception Handling ===")
    
    # Division by zero example
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        result = num1 / num2
        print(f"Result: {result}")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
    except ValueError:
        print("Error: Please enter valid numbers!")
    except Exception as e:
        print(f"Unexpected error: {e}")
    else:
        print("Division completed successfully")
    finally:
        print("Division operation finished")

# Multiple exception types
def input_validation_system():
    """System with comprehensive input validation"""
    print("\n=== Student Registration System ===")
    
    def get_student_age():
        while True:
            try:
                age = int(input("Enter student age (18-100): "))
                if age < 18 or age > 100:
                    raise ValueError("Age must be between 18 and 100")
                return age
            except ValueError as e:
                if "invalid literal" in str(e):
                    print("Error: Please enter a valid number")
                else:
                    print(f"Error: {e}")
    
    def get_student_email():
        while True:
            try:
                email = input("Enter email address: ")
                if "@" not in email or "." not in email:
                    raise ValueError("Invalid email format")
                return email
            except ValueError as e:
                print(f"Error: {e}")
    
    def get_programming_experience():
        while True:
            try:
                experience = float(input("Years of programming experience (0-20): "))
                if experience < 0 or experience > 20:
                    raise ValueError("Experience must be between 0-20 years")
                return experience
            except ValueError as e:
                if "invalid literal" in str(e):
                    print("Error: Please enter a valid number")
                else:
                    print(f"Error: {e}")
    
    # Collect student information with error handling
    try:
        name = input("Enter student name: ")
        if not name.strip():
            raise ValueError("Name cannot be empty")
        
        age = get_student_age()
        email = get_student_email()
        experience = get_programming_experience()
        
        # Create student record
        student_record = {
            "name": name,
            "age": age,
            "email": email,
            "experience": experience,
            "registration_date": "2025-09-10"
        }
        
        print(f"\nStudent registered successfully!")
        print(f"Name: {student_record['name']}")
        print(f"Age: {student_record['age']}")
        print(f"Email: {student_record['email']}")
        print(f"Experience: {student_record['experience']} years")
        
        return student_record
        
    except KeyboardInterrupt:
        print("\nRegistration cancelled by user")
        return None
    except Exception as e:
        print(f"Registration failed: {e}")
        return None

# Custom exceptions
class LearningError(Exception):
    """Custom exception for learning-related errors"""
    pass

class SkillLevelError(LearningError):
    """Exception for skill level issues"""
    pass

class ProgressError(LearningError):
    """Exception for progress tracking issues"""
    pass

class AIMLLearningSystem:
    """AI/ML Learning system with custom exception handling"""
    
    def __init__(self, student_name):
        self.student_name = student_name
        self.skills = {}
        self.current_day = 1
        self.max_skills_per_day = 5
    
    def add_skill(self, skill_name, difficulty_level):
        """Add skill with difficulty validation"""
        try:
            if difficulty_level not in [1, 2, 3, 4, 5]:
                raise SkillLevelError(f"Difficulty level must be 1-5, got {difficulty_level}")
            
            if len(self.skills) >= self.max_skills_per_day * self.current_day:
                raise ProgressError(f"Too many skills for day {self.current_day}. Max: {self.max_skills_per_day}")
            
            if skill_name in self.skills:
                raise LearningError(f"Skill '{skill_name}' already exists")
            
            self.skills[skill_name] = {
                'difficulty': difficulty_level,
                'day_learned': self.current_day,
                'practice_hours': 0
            }
            
            print(f"Added skill: {skill_name} (Difficulty: {difficulty_level})")
            
        except SkillLevelError as e:
            print(f"Skill Level Error: {e}")
        except ProgressError as e:
            print(f"Progress Error: {e}")
        except LearningError as e:
            print(f"Learning Error: {e}")
    
    def practice_skill(self, skill_name, hours):
        """Practice a skill with validation"""
        try:
            if skill_name not in self.skills:
                raise LearningError(f"Skill '{skill_name}' not found. Learn it first!")
            
            if hours <= 0:
                raise ValueError("Practice hours must be positive")
            
            if hours > 8:
                raise ProgressError("Cannot practice more than 8 hours per day")
            
            self.skills[skill_name]['practice_hours'] += hours
            print(f"Practiced {skill_name} for {hours} hours")
            
        except LearningError as e:
            print(f"Learning Error: {e}")
        except ProgressError as e:
            print(f"Progress Error: {e}")
        except ValueError as e:
            print(f"Value Error: {e}")
    
    def advance_day(self):
        """Move to next day"""
        self.current_day += 1
        print(f"Advanced to Day {self.current_day}")
    
    def get_skill_summary(self):
        """Get summary of all skills"""
        try:
            if not self.skills:
                raise LearningError("No skills learned yet")
            
            print(f"\n=== Skill Summary for {self.student_name} ===")
            total_hours = 0
            for skill, data in self.skills.items():
                print(f"{skill}: Difficulty {data['difficulty']}, "
                      f"Learned Day {data['day_learned']}, "
                      f"Practice: {data['practice_hours']}h")
                total_hours += data['practice_hours']
            
            print(f"Total Practice Hours: {total_hours}")
            print(f"Average Hours per Skill: {total_hours/len(self.skills):.1f}")
            
        except LearningError as e:
            print(f"Learning Error: {e}")

# Context managers (with statement)
class LearningSession:
    """Context manager for learning sessions"""
    
    def __init__(self, topic, duration):
        self.topic = topic
        self.duration = duration
        self.start_time = None
    
    def __enter__(self):
        from datetime import datetime
        self.start_time = datetime.now()
        print(f"Starting learning session: {self.topic}")
        print(f"Planned duration: {self.duration} hours")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from datetime import datetime
        end_time = datetime.now()
        actual_duration = (end_time - self.start_time).seconds / 3600
        
        if exc_type is None:
            print(f"Learning session completed successfully!")
            print(f"Topic: {self.topic}")
            print(f"Actual duration: {actual_duration:.2f} hours")
        else:
            print(f"Learning session interrupted: {exc_val}")
        
        print(f"Session ended at: {end_time.strftime('%H:%M:%S')}")
        return False  # Don't suppress exceptions

# File handling with context manager
def safe_data_processing():
    """Demonstrate safe file processing with context managers"""
    
    try:
        with open("student_data.txt", "w") as file:
            file.write("Student Learning Data\n")
            file.write("====================\n")
            file.write("Day 3: Object-Oriented Programming\n")
            file.write("Exception Handling Completed\n")
        
        with open("student_data.txt", "r") as file:
            content = file.read()
            print("=== File Content ===")
            print(content)
            
    except IOError as e:
        print(f"File operation failed: {e}")

# Testing all exception handling
if __name__ == "__main__":
    print("Testing exception handling systems...")
    
    # Basic exceptions
    print("1. Basic Exception Demo:")
    # basic_exception_demo()  # Commented out for automated testing
    
    # Student registration with validation
    print("\n2. Student Registration System:")
    # student_record = input_validation_system()  # Commented out for automated testing
    
    # Custom exceptions
    print("\n3. Custom Exception Demo:")
    learning_system = AIMLLearningSystem("Ashwin")
    
    # Test custom exceptions
    learning_system.add_skill("Python Basics", 2)
    learning_system.add_skill("OOP", 3)
    learning_system.add_skill("Exception Handling", 4)
    learning_system.practice_skill("Python Basics", 2)
    learning_system.practice_skill("Advanced AI", 1)  # This will raise an error
    learning_system.practice_skill("Python Basics", -1)  # This will raise an error
    learning_system.get_skill_summary()
    
    # Context manager demo
    print("\n4. Context Manager Demo:")
    try:
        with LearningSession("Exception Handling", 1) as session:
            print("Learning about try-except blocks...")
            print("Learning about custom exceptions...")
            print("Learning about context managers...")
            # Simulate some work
            import time
            time.sleep(1)
    except Exception as e:
        print(f"Session error: {e}")
    
    # Safe file processing
    print("\n5. Safe File Processing:")
    safe_data_processing()
    
    print("\nException handling practice completed!")