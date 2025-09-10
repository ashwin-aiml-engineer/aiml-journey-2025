import os
import json
from datetime import datetime

print("=== File Handling Practice ===")

# Basic file writing
def create_learning_log():
    """Create a learning log file"""
    log_content = """AI/ML Learning Journey - Daily Log
=====================================
Day 1: Python Basics - Variables, Operators, I/O
Day 2: Control Structures, Loops, Functions
Day 3: Object-Oriented Programming, File Handling

Skills Acquired:
- Python syntax fundamentals
- Problem-solving with code
- Object-oriented thinking
- File operations

Next Goals:
- Data structures (lists, dictionaries)
- Libraries (NumPy, Pandas)
- Machine learning basics
"""
    
    # Writing to file
    with open("learning_log.txt", "w") as file:
        file.write(log_content)
    print("Created learning_log.txt")

# Reading files
def read_learning_log():
    """Read and display learning log"""
    try:
        with open("learning_log.txt", "r") as file:
            content = file.read()
            print("=== Learning Log Content ===")
            print(content)
    except FileNotFoundError:
        print("Learning log not found. Creating it first...")
        create_learning_log()
        read_learning_log()

# Appending to files
def update_daily_progress(day, topics, hours):
    """Append daily progress to log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_entry = f"\n[{timestamp}] Day {day}: {', '.join(topics)} - {hours} hours studied"
    
    with open("daily_progress.txt", "a") as file:
        file.write(progress_entry)
    print(f"Added Day {day} progress to log")

# Working with CSV files
def create_skills_database():
    """Create a CSV file to track skills"""
    import csv
    
    skills_data = [
        ["Skill", "Level", "Days_Practiced", "Projects_Used"],
        ["Python Basics", "Intermediate", "3", "5"],
        ["Control Structures", "Intermediate", "2", "3"],
        ["Functions", "Beginner", "2", "4"],
        ["OOP", "Beginner", "1", "1"],
        ["File Handling", "Beginner", "1", "1"]
    ]
    
    with open("skills_database.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(skills_data)
    print("Created skills_database.csv")

def read_skills_database():
    """Read and analyze skills database"""
    import csv
    
    try:
        with open("skills_database.csv", "r") as file:
            reader = csv.DictReader(file)
            print("\n=== Current Skills Database ===")
            for row in reader:
                print(f"{row['Skill']}: Level {row['Level']} - {row['Days_Practiced']} days practice")
    except FileNotFoundError:
        print("Skills database not found. Creating it...")
        create_skills_database()
        read_skills_database()

# Working with JSON files
def create_project_portfolio():
    """Create JSON file for project portfolio"""
    portfolio = {
        "student_name": "Ashwin Shetty",
        "program": "AI/ML Engineering Bootcamp",
        "start_date": "2025-09-08",
        "current_day": 3,
        "projects": [
            {
                "name": "Hello World Program",
                "day": 1,
                "technologies": ["Python"],
                "description": "First Python program with variables and I/O",
                "github_link": "day_01_hello_world.py"
            },
            {
                "name": "Control Structures Practice",
                "day": 2,
                "technologies": ["Python", "Conditional Logic"],
                "description": "Decision making and flow control",
                "github_link": "day_02_control_structures.py"
            },
            {
                "name": "OOP Learning System",
                "day": 3,
                "technologies": ["Python", "OOP", "Classes"],
                "description": "Student management with object-oriented design",
                "github_link": "day_03_oop_basics.py"
            }
        ],
        "skills": {
            "programming": ["Python", "Git", "VS Code"],
            "concepts": ["Variables", "Functions", "OOP", "File Handling"],
            "tools": ["GitHub", "Command Line"]
        },
        "metrics": {
            "total_hours": 9,
            "lines_of_code": 500,
            "files_created": 8,
            "github_commits": 6
        }
    }
    
    with open("project_portfolio.json", "w") as file:
        json.dump(portfolio, file, indent=2)
    print("Created project_portfolio.json")

def read_project_portfolio():
    """Read and display project portfolio"""
    try:
        with open("project_portfolio.json", "r") as file:
            portfolio = json.load(file)
            
            print("\n=== Project Portfolio ===")
            print(f"Student: {portfolio['student_name']}")
            print(f"Program: {portfolio['program']}")
            print(f"Current Day: {portfolio['current_day']}")
            
            print(f"\nProjects ({len(portfolio['projects'])}):")
            for project in portfolio['projects']:
                print(f"  - {project['name']} (Day {project['day']})")
                print(f"    Technologies: {', '.join(project['technologies'])}")
            
            print(f"\nTotal Skills: {len(portfolio['skills']['programming']) + len(portfolio['skills']['concepts'])}")
            print(f"Total Hours: {portfolio['metrics']['total_hours']}")
            print(f"Lines of Code: {portfolio['metrics']['lines_of_code']}")
            
    except FileNotFoundError:
        print("Portfolio not found. Creating it...")
        create_project_portfolio()
        read_project_portfolio()

# File system operations
def organize_learning_files():
    """Create organized folder structure for learning materials"""
    folders = [
        "daily_code",
        "projects", 
        "notes",
        "resources",
        "data_files"
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

def backup_progress():
    """Create backup of all progress files"""
    import shutil
    from datetime import datetime
    
    backup_folder = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    files_to_backup = [
        "learning_log.txt",
        "daily_progress.txt", 
        "skills_database.csv",
        "project_portfolio.json"
    ]
    
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    backed_up_files = 0
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_folder)
            backed_up_files += 1
            print(f"Backed up: {file}")
    
    print(f"Backup completed: {backed_up_files} files in {backup_folder}")

# Exception handling with files
def safe_file_operations():
    """Demonstrate safe file operations with error handling"""
    
    def safe_read_file(filename):
        try:
            with open(filename, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return None
        except PermissionError:
            print(f"Error: No permission to read '{filename}'")
            return None
        except Exception as e:
            print(f"Unexpected error reading '{filename}': {e}")
            return None
    
    def safe_write_file(filename, content):
        try:
            with open(filename, 'w') as file:
                file.write(content)
            print(f"Successfully wrote to '{filename}'")
            return True
        except PermissionError:
            print(f"Error: No permission to write '{filename}'")
            return False
        except Exception as e:
            print(f"Unexpected error writing '{filename}': {e}")
            return False
    
    # Test safe operations
    content = safe_read_file("learning_log.txt")
    if content:
        print("File read successfully")
    
    test_content = "Test file content for error handling practice"
    safe_write_file("test_file.txt", test_content)

# Run all file operations
if __name__ == "__main__":
    print("Starting file handling demonstrations...")
    
    # Basic file operations
    create_learning_log()
    read_learning_log()
    
    # Progress tracking
    update_daily_progress(1, ["Variables", "Operators"], 3)
    update_daily_progress(2, ["Control Structures", "Loops"], 3)
    update_daily_progress(3, ["OOP", "File Handling"], 3)
    
    # Database operations
    create_skills_database()
    read_skills_database()
    
    # JSON portfolio
    create_project_portfolio()
    read_project_portfolio()
    
    # File organization
    organize_learning_files()
    
    # Safe operations
    safe_file_operations()
    
    # Backup
    backup_progress()
    
    print("\nFile handling practice completed!")