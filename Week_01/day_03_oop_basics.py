print("=== Object-Oriented Programming Basics ===")

# Basic class definition
class Student:
    """A class representing a student in AI/ML program"""
    
    # Class variable (shared by all instances)
    program_name = "AI/ML Engineering"
    
    def __init__(self, name, age, background):
        """Constructor method - called when object is created"""
        self.name = name  # Instance variable
        self.age = age
        self.background = background
        self.skills = []
        self.projects = []
        self.day = 1
    
    def introduce(self):
        """Method to introduce the student"""
        print(f"Hi, I'm {self.name}, {self.age} years old")
        print(f"Background: {self.background}")
        print(f"Learning: {self.program_name}")
    
    def add_skill(self, skill):
        """Method to add a new skill"""
        if skill not in self.skills:
            self.skills.append(skill)
            print(f"Added skill: {skill}")
        else:
            print(f"Already have skill: {skill}")
    
    def complete_day(self, topics_covered):
        """Mark completion of a learning day"""
        self.day += 1
        print(f"Day {self.day-1} completed! Topics: {', '.join(topics_covered)}")
        return self.day - 1
    
    def show_progress(self):
        """Display current progress"""
        print(f"\n=== Progress Report for {self.name} ===")
        print(f"Current Day: {self.day}")
        print(f"Skills Learned: {', '.join(self.skills) if self.skills else 'None yet'}")
        print(f"Total Skills: {len(self.skills)}")
        completion = (self.day / 90) * 100
        print(f"Program Completion: {completion:.1f}%")

# Create student objects
student1 = Student("Ashwin", 23, "Mechanical Engineering")
student2 = Student("Priya", 23, "Electrical Engineering")

# Use methods
student1.introduce()
student1.add_skill("Python Basics")
student1.add_skill("Control Structures")
student1.add_skill("Functions")
student1.complete_day(["OOP", "Classes", "Objects"])
student1.show_progress()

print("\n" + "="*50)

# Class with inheritance
class AIMLStudent(Student):
    """Specialized class for AI/ML students - inherits from Student"""
    
    def __init__(self, name, age, background, specialization):
        super().__init__(name, age, background)  # Call parent constructor
        self.specialization = specialization
        self.ml_projects = []
        self.certifications = []
    
    def add_ml_project(self, project_name, technologies):
        """Add a machine learning project"""
        project = {
            'name': project_name,
            'technologies': technologies,
            'day_completed': self.day
        }
        self.ml_projects.append(project)
        print(f"Added ML project: {project_name}")
    
    def earn_certification(self, cert_name):
        """Add a certification"""
        self.certifications.append(cert_name)
        print(f"Earned certification: {cert_name}")
    
    def show_ml_progress(self):
        """Display AI/ML specific progress"""
        print(f"\n=== AI/ML Progress for {self.name} ===")
        print(f"Specialization: {self.specialization}")
        print(f"ML Projects: {len(self.ml_projects)}")
        for project in self.ml_projects:
            print(f"  - {project['name']} (Day {project['day_completed']})")
        print(f"Certifications: {len(self.certifications)}")
        for cert in self.certifications:
            print(f"  - {cert}")

# Create specialized student
ml_student = AIMLStudent("Ashwin", 25, "Mechanical Engineering", "Computer Vision")
ml_student.add_skill("Python")
ml_student.add_skill("Machine Learning")
ml_student.add_ml_project("Image Classifier", ["Python", "TensorFlow"])
ml_student.earn_certification("Python Fundamentals")
ml_student.show_ml_progress()

# Class with encapsulation (private attributes)
class LearningTracker:
    """Class to track learning progress with private data"""
    
    def __init__(self, student_name):
        self.student_name = student_name
        self.__daily_hours = []  # Private attribute 
        self.__total_hours = 0
        self._streak_days = 0  # Protected attribute
    
    def log_daily_study(self, hours, topics):
        """Log daily study session"""
        if hours > 0:
            self.__daily_hours.append(hours)
            self.__total_hours += hours
            self._streak_days += 1
            print(f"Logged {hours} hours studying: {', '.join(topics)}")
        else:
            print("Study hours must be positive")
    
    def get_total_hours(self):
        """Getter method for total hours"""
        return self.__total_hours
    
    def get_average_daily_hours(self):
        """Calculate average daily study hours"""
        if len(self.__daily_hours) > 0:
            return self.__total_hours / len(self.__daily_hours)
        return 0
    
    def get_streak(self):
        """Get current learning streak"""
        return self._streak_days
    
    def display_stats(self):
        """Display learning statistics"""
        print(f"\n=== Learning Stats for {self.student_name} ===")
        print(f"Total Study Hours: {self.__total_hours}")
        print(f"Study Days: {len(self.__daily_hours)}")
        print(f"Average Daily Hours: {self.get_average_daily_hours():.1f}")
        print(f"Learning Streak: {self._streak_days} days")

# Test encapsulation
tracker = LearningTracker("Ashwin")
tracker.log_daily_study(3, ["Variables", "Operators"])
tracker.log_daily_study(3, ["Control Structures", "Loops"])
tracker.log_daily_study(3, ["Functions", "OOP"])
tracker.display_stats()

# Class methods and static methods
class ProgrammingUtils:
    """Utility class with various helper methods"""
    
    supported_languages = ["Python", "Java", "C++", "JavaScript"]
    
    def __init__(self, language):
        self.language = language
    
    @classmethod
    def add_language(cls, language):
        """Class method to add supported language"""
        if language not in cls.supported_languages:
            cls.supported_languages.append(language)
            print(f"Added {language} to supported languages")
    
    @staticmethod
    def calculate_learning_time(skill_level, target_level, hours_per_day):
        """Static method - doesn't need instance or class"""
        skill_gap = target_level - skill_level
        days_needed = (skill_gap * 20) / hours_per_day
        return max(1, int(days_needed))
    
    def practice_syntax(self):
        """Instance method"""
        print(f"Practicing {self.language} syntax...")

# Test class and static methods
ProgrammingUtils.add_language("R")
time_needed = ProgrammingUtils.calculate_learning_time(2, 8, 3)
print(f"Estimated learning time: {time_needed} days")

python_learner = ProgrammingUtils("Python")
python_learner.practice_syntax()