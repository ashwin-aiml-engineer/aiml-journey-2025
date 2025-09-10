import json
import os
import csv
from datetime import datetime, timedelta

class Student:
    """Student class with learning tracking capabilities"""
    
    def __init__(self, student_id, name, email, background):
        self.student_id = student_id
        self.name = name
        self.email = email
        self.background = background
        self.enrollment_date = datetime.now()
        self.current_day = 1
        self.completed_courses = []
        self.skills = {}
        self.total_study_hours = 0
        self.certificates = []
        self.learning_streak = 0
    
    def add_skill(self, skill_name, proficiency_level):
        """Add or update a skill"""
        if skill_name in self.skills:
            old_level = self.skills[skill_name]['level']
            self.skills[skill_name] = {
                'level': proficiency_level,
                'last_updated': datetime.now().isoformat(),
                'previous_level': old_level
            }
            print(f"Updated {skill_name}: {old_level} -> {proficiency_level}")
        else:
            self.skills[skill_name] = {
                'level': proficiency_level,
                'date_learned': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            print(f"New skill added: {skill_name} (Level: {proficiency_level})")
    
    def complete_course(self, course_title, hours_spent, grade):
        """Mark a course as completed"""
        completion_record = {
            'title': course_title,
            'completion_date': datetime.now().isoformat(),
            'hours_spent': hours_spent,
            'grade': grade,
            'day_completed': self.current_day
        }
        self.completed_courses.append(completion_record)
        self.total_study_hours += hours_spent
        
        if grade >= 80:
            certificate = f"{course_title} Certificate - Grade: {grade}%"
            self.certificates.append(certificate)
            print(f"Congratulations! Certificate earned: {certificate}")
    
    def advance_day(self):
        """Move to next learning day"""
        self.current_day += 1
        self.learning_streak += 1
    
    def get_progress_summary(self):
        """Generate comprehensive progress summary"""
        return {
            'student_id': self.student_id,
            'name': self.name,
            'current_day': self.current_day,
            'total_skills': len(self.skills),
            'completed_courses': len(self.completed_courses),
            'total_hours': self.total_study_hours,
            'certificates': len(self.certificates),
            'learning_streak': self.learning_streak,
            'enrollment_duration': (datetime.now() - self.enrollment_date).days
        }
    
    def to_dict(self):
        """Convert student object to dictionary for JSON storage"""
        return {
            'student_id': self.student_id,
            'name': self.name,
            'email': self.email,
            'background': self.background,
            'enrollment_date': self.enrollment_date.isoformat(),
            'current_day': self.current_day,
            'completed_courses': self.completed_courses,
            'skills': self.skills,
            'total_study_hours': self.total_study_hours,
            'certificates': self.certificates,
            'learning_streak': self.learning_streak
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create student object from dictionary"""
        student = cls(
            data['student_id'],
            data['name'],
            data['email'],
            data['background']
        )
        student.enrollment_date = datetime.fromisoformat(data['enrollment_date'])
        student.current_day = data['current_day']
        student.completed_courses = data['completed_courses']
        student.skills = data['skills']
        student.total_study_hours = data['total_study_hours']
        student.certificates = data['certificates']
        student.learning_streak = data['learning_streak']
        return student

class Course:
    """Course class with detailed information"""
    
    def __init__(self, course_id, title, description, difficulty, estimated_hours):
        self.course_id = course_id
        self.title = title
        self.description = description
        self.difficulty = difficulty  # Beginner, Intermediate, Advanced
        self.estimated_hours = estimated_hours
        self.enrolled_students = []
        self.completion_rate = 0.0
        self.average_grade = 0.0
    
    def enroll_student(self, student_id):
        """Enroll a student in the course"""
        if student_id not in self.enrolled_students:
            self.enrolled_students.append(student_id)
            return True
        return False
    
    def to_dict(self):
        """Convert course to dictionary"""
        return {
            'course_id': self.course_id,
            'title': self.title,
            'description': self.description,
            'difficulty': self.difficulty,
            'estimated_hours': self.estimated_hours,
            'enrolled_students': self.enrolled_students,
            'completion_rate': self.completion_rate,
            'average_grade': self.average_grade
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create course from dictionary"""
        course = cls(
            data['course_id'],
            data['title'],
            data['description'],
            data['difficulty'],
            data['estimated_hours']
        )
        course.enrolled_students = data['enrolled_students']
        course.completion_rate = data['completion_rate']
        course.average_grade = data['average_grade']
        return course

class LearningManagementSystem:
    """Complete learning management system"""
    
    def __init__(self, data_directory="learning_data"):
        self.data_directory = data_directory
        self.students = {}
        self.courses = {}
        self.analytics = {}
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        self.students_file = os.path.join(self.data_directory, "students.json")
        self.courses_file = os.path.join(self.data_directory, "courses.json")
        self.analytics_file = os.path.join(self.data_directory, "analytics.json")
        
        self.load_all_data()
    
    def load_all_data(self):
        """Load all data from files"""
        try:
            self.load_students()
            self.load_courses()
            self.load_analytics()
            print("All data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Starting with fresh data")
    
    def load_students(self):
        """Load students from JSON file"""
        try:
            with open(self.students_file, 'r') as file:
                students_data = json.load(file)
                for student_id, data in students_data.items():
                    self.students[student_id] = Student.from_dict(data)
        except FileNotFoundError:
            print("No existing student data found")
        except json.JSONDecodeError:
            print("Error reading student data file")
    
    def load_courses(self):
        """Load courses from JSON file"""
        try:
            with open(self.courses_file, 'r') as file:
                courses_data = json.load(file)
                for course_id, data in courses_data.items():
                    self.courses[course_id] = Course.from_dict(data)
        except FileNotFoundError:
            print("No existing course data found")
        except json.JSONDecodeError:
            print("Error reading course data file")
    
    def load_analytics(self):
        """Load analytics from JSON file"""
        try:
            with open(self.analytics_file, 'r') as file:
                self.analytics = json.load(file)
        except FileNotFoundError:
            self.analytics = {
                'total_students': 0,
                'total_courses': 0,
                'total_completions': 0,
                'average_study_hours': 0.0
            }
    
    def save_all_data(self):
        """Save all data to files"""
        try:
            self.save_students()
            self.save_courses()
            self.save_analytics()
            print("All data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def save_students(self):
        """Save students to JSON file"""
        students_data = {}
        for student_id, student in self.students.items():
            students_data[student_id] = student.to_dict()
        
        with open(self.students_file, 'w') as file:
            json.dump(students_data, file, indent=2, default=str)
    
    def save_courses(self):
        """Save courses to JSON file"""
        courses_data = {}
        for course_id, course in self.courses.items():
            courses_data[course_id] = course.to_dict()
        
        with open(self.courses_file, 'w') as file:
            json.dump(courses_data, file, indent=2)
    
    def save_analytics(self):
        """Save analytics to JSON file"""
        with open(self.analytics_file, 'w') as file:
            json.dump(self.analytics, file, indent=2)
    
    def add_student(self, student_id, name, email, background):
        """Add new student with validation"""
        if student_id in self.students:
            raise ValueError(f"Student with ID {student_id} already exists")
        
        if not name.strip():
            raise ValueError("Student name cannot be empty")
        
        if "@" not in email or "." not in email:
            raise ValueError("Invalid email format")
        
        student = Student(student_id, name, email, background)
        self.students[student_id] = student
        self.analytics['total_students'] += 1
        print(f"Student added: {name} ({student_id})")
        return student
    
    def add_course(self, course_id, title, description, difficulty, estimated_hours):
        """Add new course with validation"""
        if course_id in self.courses:
            raise ValueError(f"Course with ID {course_id} already exists")
        
        valid_difficulties = ['Beginner', 'Intermediate', 'Advanced']
        if difficulty not in valid_difficulties:
            raise ValueError(f"Difficulty must be one of: {valid_difficulties}")
        
        if estimated_hours <= 0:
            raise ValueError("Estimated hours must be positive")
        
        course = Course(course_id, title, description, difficulty, estimated_hours)
        self.courses[course_id] = course
        self.analytics['total_courses'] += 1
        print(f"Course added: {title} ({course_id})")
        return course
    
    def enroll_student_in_course(self, student_id, course_id):
        """Enroll student in course"""
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        if course_id not in self.courses:
            raise ValueError(f"Course {course_id} not found")
        
        student = self.students[student_id]
        course = self.courses[course_id]
        
        if course.enroll_student(student_id):
            print(f"{student.name} enrolled in {course.title}")
        else:
            print(f"{student.name} is already enrolled in {course.title}")
    
    def complete_course(self, student_id, course_id, hours_spent, grade):
        """Mark course completion for student"""
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        if course_id not in self.courses:
            raise ValueError(f"Course {course_id} not found")
        
        if grade < 0 or grade > 100:
            raise ValueError("Grade must be between 0 and 100")
        
        student = self.students[student_id]
        course = self.courses[course_id]
        
        student.complete_course(course.title, hours_spent, grade)
        self.analytics['total_completions'] += 1
        
        # Update course statistics
        completed_students = sum(1 for s in self.students.values() 
                               if any(c['title'] == course.title for c in s.completed_courses))
        course.completion_rate = (completed_students / len(course.enrolled_students) * 100) if course.enrolled_students else 0
        
        print(f"{student.name} completed {course.title} with grade {grade}%")
    
    def generate_student_report(self, student_id):
        """Generate comprehensive student report"""
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        student = self.students[student_id]
        summary = student.get_progress_summary()
        
        report = f"""
=== STUDENT PROGRESS REPORT ===
Name: {student.name}
Student ID: {student.student_id}
Email: {student.email}
Background: {student.background}
Enrollment Date: {student.enrollment_date.strftime('%Y-%m-%d')}

LEARNING PROGRESS:
Current Day: {summary['current_day']}
Learning Streak: {summary['learning_streak']} days
Total Study Hours: {summary['total_hours']}
Enrollment Duration: {summary['enrollment_duration']} days

SKILLS ACQUIRED: {summary['total_skills']}
"""
        
        if student.skills:
            report += "Detailed Skills:\n"
            for skill, data in student.skills.items():
                report += f"  - {skill}: Level {data['level']}\n"
        
        report += f"\nCOURSES COMPLETED: {summary['completed_courses']}\n"
        if student.completed_courses:
            for course in student.completed_courses:
                report += f"  - {course['title']}: {course['grade']}% (Day {course['day_completed']})\n"
        
        report += f"\nCERTIFICATES EARNED: {summary['certificates']}\n"
        if student.certificates:
            for cert in student.certificates:
                report += f"  - {cert}\n"
        
        return report
    
    def export_data_to_csv(self):
        """Export student data to CSV files"""
        # Export students
        students_csv = os.path.join(self.data_directory, "students_export.csv")
        with open(students_csv, 'w', newline='') as csvfile:
            fieldnames = ['student_id', 'name', 'email', 'background', 'current_day', 
                         'total_hours', 'completed_courses', 'total_skills', 'certificates']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for student in self.students.values():
                summary = student.get_progress_summary()
                writer.writerow({
                    'student_id': student.student_id,
                    'name': student.name,
                    'email': student.email,
                    'background': student.background,
                    'current_day': student.current_day,
                    'total_hours': student.total_study_hours,
                    'completed_courses': len(student.completed_courses),
                    'total_skills': len(student.skills),
                    'certificates': len(student.certificates)
                })
        
        print(f"Student data exported to {students_csv}")
        
        # Export courses
        courses_csv = os.path.join(self.data_directory, "courses_export.csv")
        with open(courses_csv, 'w', newline='') as csvfile:
            fieldnames = ['course_id', 'title', 'difficulty', 'estimated_hours', 
                         'enrolled_students', 'completion_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for course in self.courses.values():
                writer.writerow({
                    'course_id': course.course_id,
                    'title': course.title,
                    'difficulty': course.difficulty,
                    'estimated_hours': course.estimated_hours,
                    'enrolled_students': len(course.enrolled_students),
                    'completion_rate': course.completion_rate
                })
        
        print(f"Course data exported to {courses_csv}")
    
    def get_system_analytics(self):
        """Generate system-wide analytics"""
        total_hours = sum(student.total_study_hours for student in self.students.values())
        avg_hours = total_hours / len(self.students) if self.students else 0
        
        self.analytics.update({
            'total_students': len(self.students),
            'total_courses': len(self.courses),
            'total_study_hours': total_hours,
            'average_study_hours': avg_hours,
            'last_updated': datetime.now().isoformat()
        })
        
        return self.analytics

def demonstrate_learning_system():
    """Demonstrate the complete learning management system"""
    print("=== LEARNING MANAGEMENT SYSTEM DEMO ===\n")
    
    # Initialize system
    lms = LearningManagementSystem()
    
    try:
        # Add courses
        print("Adding courses...")
        lms.add_course("PY101", "Python Fundamentals", 
                      "Complete Python basics including variables, control structures, functions", 
                      "Beginner", 25)
        
        lms.add_course("OOP101", "Object-Oriented Programming", 
                      "Classes, objects, inheritance, and encapsulation in Python", 
                      "Intermediate", 20)
        
        lms.add_course("FH101", "File Handling & Data Management", 
                      "Working with files, JSON, CSV, and data persistence", 
                      "Intermediate", 15)
        
        lms.add_course("ML101", "Machine Learning Basics", 
                      "Introduction to machine learning concepts and algorithms", 
                      "Advanced", 40)
        
        print("\nAdding students...")
        # Add students
        lms.add_student("STU001", "Ashwin Shetty", "ashwin@email.com", "Mechanical Engineering")
        lms.add_student("STU002", "Priya Sharma", "priya@email.com", "Computer Science")
        lms.add_student("STU003", "Rahul Kumar", "rahul@email.com", "Electrical Engineering")
        
        print("\nEnrolling students in courses...")
        # Enroll students
        lms.enroll_student_in_course("STU001", "PY101")
        lms.enroll_student_in_course("STU001", "OOP101")
        lms.enroll_student_in_course("STU001", "FH101")
        
        lms.enroll_student_in_course("STU002", "PY101")
        lms.enroll_student_in_course("STU002", "ML101")
        
        print("\nAdding skills to students...")
        # Add skills
        ashwin = lms.students["STU001"]
        ashwin.add_skill("Python Programming", 7)
        ashwin.add_skill("Object-Oriented Design", 6)
        ashwin.add_skill("File Operations", 5)
        ashwin.add_skill("Problem Solving", 8)
        
        priya = lms.students["STU002"]
        priya.add_skill("Python Programming", 8)
        priya.add_skill("Machine Learning", 6)
        priya.add_skill("Data Analysis", 7)
        
        print("\nCompleting courses...")
        # Complete courses
        lms.complete_course("STU001", "PY101", 20, 85)
        lms.complete_course("STU001", "OOP101", 18, 88)
        lms.complete_course("STU001", "FH101", 12, 82)
        
        lms.complete_course("STU002", "PY101", 22, 92)
        
        print("\nAdvancing learning days...")
        # Advance days
        ashwin.advance_day()
        ashwin.advance_day()
        ashwin.advance_day()
        
        priya.advance_day()
        priya.advance_day()
        
        print("\n" + "="*60)
        print("GENERATING REPORTS...")
        print("="*60)
        
        # Generate reports
        ashwin_report = lms.generate_student_report("STU001")
        print(ashwin_report)
        
        priya_report = lms.generate_student_report("STU002")
        print(priya_report)
        
        # System analytics
        print("\n=== SYSTEM ANALYTICS ===")
        analytics = lms.get_system_analytics()
        for key, value in analytics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\nExporting data...")
        # Export data
        lms.export_data_to_csv()
        
        print("\nSaving all data...")
        # Save all data
        lms.save_all_data()
        
        print("\n=== DEMO COMPLETED SUCCESSFULLY ===")
        
    except ValueError as e:
        print(f"Validation Error: {e}")
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    demonstrate_learning_system()