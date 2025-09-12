# Day 5: Pandas Basics - Data Manipulation and Analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== Pandas Basics for Data Analysis ===")

# CREATING PANDAS OBJECTS
print("\n1. CREATING PANDAS OBJECTS")
print("-" * 40)

# Creating Series (1D data structure)
series_from_list = pd.Series([10, 20, 30, 40, 50])
series_with_index = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series_from_dict = pd.Series({'Alice': 85, 'Bob': 90, 'Charlie': 78})

print("Series from list:")
print(series_from_list)
print("\nSeries with custom index:")
print(series_with_index)
print("\nSeries from dictionary:")
print(series_from_dict)

# Creating DataFrames (2D data structure)
# Method 1: From dictionary
student_data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [20, 21, 19, 22],
    'Grade': [85, 90, 78, 92],
    'Subject': ['Math', 'Physics', 'Chemistry', 'Biology']
}
df_from_dict = pd.DataFrame(student_data)

# Method 2: From list of dictionaries
student_records = [
    {'Name': 'Alice', 'Age': 20, 'Grade': 85, 'Subject': 'Math'},
    {'Name': 'Bob', 'Age': 21, 'Grade': 90, 'Subject': 'Physics'},
    {'Name': 'Charlie', 'Age': 19, 'Grade': 78, 'Subject': 'Chemistry'},
    {'Name': 'Diana', 'Age': 22, 'Grade': 92, 'Subject': 'Biology'}
]
df_from_records = pd.DataFrame(student_records)

print("\nDataFrame from dictionary:")
print(df_from_dict)

# DataFrame properties
print(f"\nDataFrame info:")
print(f"Shape: {df_from_dict.shape}")
print(f"Columns: {list(df_from_dict.columns)}")
print(f"Index: {list(df_from_dict.index)}")
print(f"Data types:\n{df_from_dict.dtypes}")

# DATA LOADING AND SAVING
print("\n2. DATA LOADING AND SAVING")
print("-" * 40)

# Create sample data and save to files
sample_data = {
    'student_id': [1, 2, 3, 4, 5],
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Adams'],
    'age': [20, 21, 19, 22, 20],
    'grade': [85.5, 90.2, 78.8, 92.1, 87.3],
    'subject': ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science'],
    'enrollment_date': ['2023-09-01', '2023-09-01', '2023-09-15', '2023-09-01', '2023-09-10']
}
df_sample = pd.DataFrame(sample_data)

# Save to CSV
df_sample.to_csv('students.csv', index=False)
print("Sample data saved to 'students.csv'")

# Load from CSV
df_loaded = pd.read_csv('students.csv')
print("\nLoaded data from CSV:")
print(df_loaded.head())

# Different read options
print("\nReading with specific options:")
df_custom = pd.read_csv('students.csv', 
                       usecols=['name', 'grade', 'subject'],  # Select specific columns
                       nrows=3)                               # Read only first 3 rows
print(df_custom)

# BASIC DATA EXPLORATION
print("\n3. BASIC DATA EXPLORATION")
print("-" * 40)

# Basic information about the dataset
print("Dataset overview:")
print(f"Shape: {df_loaded.shape}")
print(f"Columns: {list(df_loaded.columns)}")

# Display methods
print("\nFirst 3 rows:")
print(df_loaded.head(3))

print("\nLast 2 rows:")
print(df_loaded.tail(2))

print("\nDataset info:")
print(df_loaded.info())

print("\nDescriptive statistics:")
print(df_loaded.describe())

# DATA SELECTION AND INDEXING
print("\n4. DATA SELECTION AND INDEXING")
print("-" * 40)

# Column selection
print("Single column selection:")
print(df_loaded['name'])

print("\nMultiple column selection:")
print(df_loaded[['name', 'grade']])

# Row selection using iloc (integer location)
print("\nRow selection using iloc:")
print("First row:")
print(df_loaded.iloc[0])

print("\nFirst 3 rows, first 2 columns:")
print(df_loaded.iloc[0:3, 0:2])

# Row selection using loc (label location)
df_with_index = df_loaded.set_index('student_id')
print("\nUsing loc with custom index:")
print(df_with_index.loc[1])  # Student with ID 1

# Boolean indexing
print("\nBoolean indexing - students with grade > 85:")
high_grades = df_loaded[df_loaded['grade'] > 85]
print(high_grades)

# Complex boolean conditions
print("\nComplex conditions - students age 20+ AND grade > 80:")
filtered_students = df_loaded[(df_loaded['age'] >= 20) & (df_loaded['grade'] > 80)]
print(filtered_students[['name', 'age', 'grade']])

# DATA CLEANING
print("\n5. DATA CLEANING")
print("-" * 40)

# Create data with missing values
messy_data = {
    'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
    'age': [20, 21, 19, None, 20],
    'grade': [85.5, None, 78.8, 92.1, 87.3],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', None, 'eve@email.com']
}
df_messy = pd.DataFrame(messy_data)
print("Dataset with missing values:")
print(df_messy)

# Check for missing values
print("\nMissing values count:")
print(df_messy.isnull().sum())

print("\nMissing values as percentage:")
print((df_messy.isnull().sum() / len(df_messy)) * 100)

# Handle missing values
# Method 1: Drop rows with any missing values
df_dropped = df_messy.dropna()
print("\nAfter dropping rows with missing values:")
print(df_dropped)

# Method 2: Fill missing values
df_filled = df_messy.copy()
df_filled['name'].fillna('Unknown', inplace=True)
df_filled['age'].fillna(df_filled['age'].mean(), inplace=True)
df_filled['grade'].fillna(df_filled['grade'].median(), inplace=True)
df_filled['email'].fillna('no-email@domain.com', inplace=True)

print("\nAfter filling missing values:")
print(df_filled)

# DATA MANIPULATION
print("\n6. DATA MANIPULATION")
print("-" * 40)

# Adding new columns
df_work = df_loaded.copy()
df_work['grade_letter'] = df_work['grade'].apply(
    lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C' if x >= 70 else 'F'
)
df_work['is_adult'] = df_work['age'] >= 21

print("DataFrame with new columns:")
print(df_work[['name', 'grade', 'grade_letter', 'age', 'is_adult']])

# Modifying existing columns
df_work['name_upper'] = df_work['name'].str.upper()
df_work['subject_short'] = df_work['subject'].str[:4]  # First 4 characters

print("\nModified columns:")
print(df_work[['name', 'name_upper', 'subject', 'subject_short']])

# Sorting data
print("\nSorted by grade (descending):")
df_sorted = df_work.sort_values('grade', ascending=False)
print(df_sorted[['name', 'grade', 'subject']])

print("\nSorted by multiple columns:")
df_multi_sort = df_work.sort_values(['grade_letter', 'age'])
print(df_multi_sort[['name', 'grade_letter', 'age']])

# GROUPING AND AGGREGATION
print("\n7. GROUPING AND AGGREGATION")
print("-" * 40)

# Create more complex dataset for grouping
course_data = {
    'student': ['Alice', 'Bob', 'Charlie', 'Diana', 'Alice', 'Bob', 'Charlie'],
    'subject': ['Math', 'Math', 'Math', 'Math', 'Physics', 'Physics', 'Physics'],
    'score': [85, 90, 78, 92, 88, 85, 82],
    'semester': ['Fall', 'Fall', 'Fall', 'Fall', 'Spring', 'Spring', 'Spring']
}
df_courses = pd.DataFrame(course_data)
print("Course data for grouping:")
print(df_courses)

# Basic grouping
print("\nAverage score by subject:")
subject_avg = df_courses.groupby('subject')['score'].mean()
print(subject_avg)

print("\nMultiple aggregations:")
subject_stats = df_courses.groupby('subject')['score'].agg(['mean', 'min', 'max', 'count'])
print(subject_stats)

# Grouping by multiple columns
print("\nGrouping by subject and semester:")
multi_group = df_courses.groupby(['subject', 'semester'])['score'].mean()
print(multi_group)

# Custom aggregation
print("\nCustom aggregation functions:")
def score_range(scores):
    return scores.max() - scores.min()

custom_agg = df_courses.groupby('subject').agg(
    average=('score', 'mean'),
    total_students=('score', 'count'),
    score_range=('score', score_range)
)
print(custom_agg)

# WORKING WITH DATES AND TIMES
print("\n8. WORKING WITH DATES AND TIMES")
print("-" * 40)

# Convert string to datetime
df_dates = df_loaded.copy()
df_dates['enrollment_date'] = pd.to_datetime(df_dates['enrollment_date'])
print("DataFrame with datetime:")
print(df_dates[['name', 'enrollment_date']])
print(f"Enrollment date data type: {df_dates['enrollment_date'].dtype}")

# Extract date components
df_dates['enrollment_year'] = df_dates['enrollment_date'].dt.year
df_dates['enrollment_month'] = df_dates['enrollment_date'].dt.month
df_dates['enrollment_day'] = df_dates['enrollment_date'].dt.day
df_dates['day_of_week'] = df_dates['enrollment_date'].dt.day_name()

print("\nDate components:")
print(df_dates[['name', 'enrollment_date', 'enrollment_month', 'day_of_week']])

# Calculate time differences
today = pd.Timestamp.now()
df_dates['days_enrolled'] = (today - df_dates['enrollment_date']).dt.days
print("\nDays since enrollment:")
print(df_dates[['name', 'enrollment_date', 'days_enrolled']])

# DATA ANALYSIS METHODS
print("\n9. DATA ANALYSIS METHODS")
print("-" * 40)

# Value counts
print("Subject distribution:")
print(df_loaded['subject'].value_counts())

# Crosstab analysis
df_analysis = df_work.copy()
print("\nCross-tabulation of grade letters by age:")
crosstab = pd.crosstab(df_analysis['age'], df_analysis['grade_letter'])
print(crosstab)

# Correlation analysis
numeric_data = df_loaded[['age', 'grade']].corr()
print("\nCorrelation matrix:")
print(numeric_data)

# Statistical analysis
print("\nDetailed statistics for grades:")
grade_stats = df_loaded['grade'].describe()
print(grade_stats)

# Percentiles
print(f"\n25th percentile: {df_loaded['grade'].quantile(0.25)}")
print(f"50th percentile (median): {df_loaded['grade'].quantile(0.5)}")
print(f"75th percentile: {df_loaded['grade'].quantile(0.75)}")

# STRING OPERATIONS
print("\n10. STRING OPERATIONS")
print("-" * 40)

# String methods
df_strings = pd.DataFrame({
    'names': ['  Alice Johnson  ', 'bob smith', 'CHARLIE BROWN', 'diana prince'],
    'emails': ['alice@gmail.com', 'BOB@YAHOO.COM', 'charlie@outlook.com', 'DIANA@GMAIL.COM']
})

print("Original string data:")
print(df_strings)

# String cleaning and transformation
df_strings['names_clean'] = df_strings['names'].str.strip().str.title()
df_strings['emails_lower'] = df_strings['emails'].str.lower()
df_strings['first_name'] = df_strings['names_clean'].str.split().str[0]
df_strings['last_name'] = df_strings['names_clean'].str.split().str[1]
df_strings['email_domain'] = df_strings['emails_lower'].str.split('@').str[1]
df_strings['email_provider'] = df_strings['email_domain'].str.split('.').str[0]

print("\nCleaned string data:")
print(df_strings)

# Advanced string operations
print("\nAdvanced string operations:")

# String length and character operations
df_strings['name_length'] = df_strings['names_clean'].str.len()
df_strings['first_name_initial'] = df_strings['first_name'].str[0]
df_strings['email_username'] = df_strings['emails_lower'].str.split('@').str[0]

print("String analysis:")
print(df_strings[['names_clean', 'name_length', 'first_name_initial', 'email_username']])

# String filtering and pattern matching
gmail_users = df_strings[df_strings['emails_lower'].str.contains('gmail')]
print("\nGmail users:")
print(gmail_users[['names_clean', 'emails_lower']])

# String replacement and formatting
df_strings['formatted_name'] = df_strings['names_clean'].str.replace(' ', '_').str.lower()
df_strings['email_masked'] = df_strings['emails_lower'].str.replace(r'(.{2}).(@.)', r'\1*\2', regex=True)

print("\nFormatted data:")
print(df_strings[['names_clean', 'formatted_name', 'emails_lower', 'email_masked']])

# PIVOTING AND RESHAPING DATA
print("\n11. PIVOTING AND RESHAPING DATA")
print("-" * 40)

# Create sample data for pivoting
sales_data = {
    'salesperson': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'Alice', 'Bob'],
    'product': ['Laptop', 'Phone', 'Laptop', 'Phone', 'Laptop', 'Phone', 'Tablet', 'Tablet'],
    'quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1', 'Q1', 'Q2', 'Q2'],
    'sales': [100, 80, 90, 85, 95, 75, 60, 70]
}
df_sales = pd.DataFrame(sales_data)
print("Sales data:")
print(df_sales)

# Pivot table
pivot_table = df_sales.pivot_table(
    values='sales', 
    index='salesperson', 
    columns='product', 
    aggfunc='sum',
    fill_value=0
)
print("\nPivot table:")
print(pivot_table)

# Pivot with multiple aggregations
pivot_multi = df_sales.pivot_table(
    values='sales',
    index='salesperson',
    columns='quarter',
    aggfunc=['sum', 'mean'],
    fill_value=0
)
print("\nPivot with multiple aggregations:")
print(pivot_multi)

# Melting (unpivoting)
melted = pivot_table.reset_index().melt(
    id_vars='salesperson',
    var_name='product',
    value_name='sales'
)
print("\nMelted data (back to long format):")
print(melted)

# Stack and unstack operations
print("\nStack and unstack operations:")
stacked = pivot_table.stack()
print("Stacked data:")
print(stacked)

unstacked = stacked.unstack()
print("\nUnstacked back:")
print(unstacked)

# COMBINING DATAFRAMES
print("\n12. COMBINING DATAFRAMES")
print("-" * 40)

# Create sample DataFrames for merging
students_df = pd.DataFrame({
    'student_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [20, 21, 19, 22, 20]
})

grades_df = pd.DataFrame({
    'student_id': [1, 2, 3, 5, 6],
    'subject': ['Math', 'Physics', 'Chemistry', 'Biology', 'History'],
    'grade': [85, 90, 78, 92, 88]
})

courses_df = pd.DataFrame({
    'student_id': [1, 1, 2, 3, 4],
    'course': ['CS101', 'MATH201', 'PHYS101', 'CHEM101', 'BIO101'],
    'credits': [3, 4, 3, 4, 3]
})

print("Students DataFrame:")
print(students_df)
print("\nGrades DataFrame:")
print(grades_df)
print("\nCourses DataFrame:")
print(courses_df)

# Different types of joins
print("\nInner join (intersection):")
inner_merged = pd.merge(students_df, grades_df, on='student_id', how='inner')
print(inner_merged)

print("\nLeft join (all students):")
left_merged = pd.merge(students_df, grades_df, on='student_id', how='left')
print(left_merged)

print("\nOuter join (all records):")
outer_merged = pd.merge(students_df, grades_df, on='student_id', how='outer')
print(outer_merged)

print("\nRight join (all grades):")
right_merged = pd.merge(students_df, grades_df, on='student_id', how='right')
print(right_merged)

# Multiple DataFrame merging
print("\nMerging multiple DataFrames:")
# First merge students and grades
temp_merge = pd.merge(students_df, grades_df, on='student_id', how='left')
# Then merge with courses
final_merge = pd.merge(temp_merge, courses_df, on='student_id', how='left')
print(final_merge)

# Concatenation
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
df3 = pd.DataFrame({'C': [13, 14, 15], 'D': [16, 17, 18]})

print("\nConcatenation examples:")
print("DataFrame 1:")
print(df1)
print("DataFrame 2:")
print(df2)

print("\nVertical concatenation:")
vertical_concat = pd.concat([df1, df2], ignore_index=True)
print(vertical_concat)

print("\nHorizontal concatenation:")
horizontal_concat = pd.concat([df1, df3], axis=1)
print(horizontal_concat)

# Append (deprecated but still useful to understand)
print("\nUsing concat instead of append:")
appended = pd.concat([df1, pd.DataFrame({'A': [99], 'B': [100]})], ignore_index=True)
print(appended)

# ADVANCED DATA ANALYSIS
print("\n13. ADVANCED DATA ANALYSIS")
print("-" * 40)

# Create comprehensive student dataset for analysis
np.random.seed(42)
advanced_data = pd.DataFrame({
    'student_id': range(1, 101),
    'name': [f'Student_{i:03d}' for i in range(1, 101)],
    'age': np.random.randint(18, 25, 100),
    'math_score': np.random.normal(75, 15, 100).clip(0, 100),
    'science_score': np.random.normal(80, 12, 100).clip(0, 100),
    'english_score': np.random.normal(78, 14, 100).clip(0, 100),
    'study_hours': np.random.randint(10, 50, 100),
    'attendance': np.random.uniform(0.7, 1.0, 100),
    'extracurricular': np.random.choice(['Sports', 'Music', 'Drama', 'Science Club', 'None'], 100)
})

print("Advanced dataset created:")
print(f"Shape: {advanced_data.shape}")
print("\nFirst 5 rows:")
print(advanced_data.head())

# Calculate derived metrics
advanced_data['average_score'] = advanced_data[['math_score', 'science_score', 'english_score']].mean(axis=1)
advanced_data['total_score'] = advanced_data[['math_score', 'science_score', 'english_score']].sum(axis=1)

# Performance categories using cut
advanced_data['performance_category'] = pd.cut(
    advanced_data['average_score'],
    bins=[0, 60, 70, 80, 90, 100],
    labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
)

# Study efficiency calculation
advanced_data['study_efficiency'] = advanced_data['average_score'] / advanced_data['study_hours']

print("\nPerformance distribution:")
print(advanced_data['performance_category'].value_counts())

print("\nTop 10 students by average score:")
top_students = advanced_data.nlargest(10, 'average_score')[['name', 'average_score', 'study_hours', 'attendance']]
print(top_students)

# Correlation analysis
correlation_columns = ['age', 'math_score', 'science_score', 'english_score', 'study_hours', 'attendance']
correlation_matrix = advanced_data[correlation_columns].corr()
print("\nCorrelation matrix:")
print(correlation_matrix.round(3))

# Advanced grouping analysis
print("\nAnalysis by performance category:")
performance_analysis = advanced_data.groupby('performance_category').agg({
    'age': ['mean', 'std'],
    'study_hours': ['mean', 'median'],
    'attendance': ['mean', 'min', 'max'],
    'study_efficiency': ['mean', 'std']
}).round(2)
print(performance_analysis)

print("\nAnalysis by extracurricular activity:")
extracurricular_analysis = advanced_data.groupby('extracurricular').agg({
    'average_score': ['mean', 'count'],
    'attendance': 'mean',
    'study_hours': 'mean'
}).round(2)
print(extracurricular_analysis)

# Cross-tabulation analysis
print("\nCross-tabulation: Performance vs Extracurricular:")
crosstab = pd.crosstab(advanced_data['performance_category'], advanced_data['extracurricular'])
print(crosstab)

# Percentages in crosstab
print("\nPercentages (row-wise):")
crosstab_pct = pd.crosstab(advanced_data['performance_category'], 
                          advanced_data['extracurricular'], 
                          normalize='index') * 100
print(crosstab_pct.round(1))

print("\n=== Pandas String Operations and Advanced Analysis Completed ===")
print("Successfully continued from the cutoff point and covered:")
print("✓ Complete string operations and cleaning")
print("✓ Advanced pivot tables and reshaping")
print("✓ Comprehensive DataFrame merging")
print("✓ Advanced statistical analysis")
print("✓ Cross-tabulation and correlation analysis")
