"""
Day 5 Mini Project: Comprehensive Student Performance Analysis System

Features:
- Loads student data from multiple sources
- Multi-subject grade analysis
- Time-series performance tracking
- Statistical summaries and visualizations prep
- Data quality validation
- Pattern/trend identification
- Automated reporting and export

Author: AI/ML Journey
"""

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING FROM MULTIPLE SOURCES
# =============================================================================

def load_student_data():
    # Simulate loading from CSV, Excel, and JSON
    # In practice, replace with pd.read_csv, pd.read_excel, pd.read_json, etc.
    np.random.seed(42)
    n = 200
    base_data = pd.DataFrame({
        'student_id': range(1, n+1),
        'name': [f"Student_{i:03d}" for i in range(1, n+1)],
        'class': np.random.choice(['10A', '10B', '10C'], n),
        'math': np.random.randint(40, 100, n),
        'science': np.random.randint(35, 100, n),
        'english': np.random.randint(30, 100, n),
        'history': np.random.randint(25, 100, n),
        'exam_date': [datetime(2025, 1, 1) + timedelta(days=int(x)) for x in np.random.randint(0, 180, n)]
    })
    # Simulate additional JSON data (attendance, activities)
    attendance_json = [
        {'student_id': i, 'attendance_pct': np.random.uniform(0.7, 1.0), 'activity': np.random.choice(['Sports', 'Music', 'None'])}
        for i in range(1, n+1)
    ]
    attendance_df = pd.DataFrame(attendance_json)
    # Merge all sources
    df = pd.merge(base_data, attendance_df, on='student_id', how='left')
    return df

student_df = load_student_data()
print(f"Loaded student data: {student_df.shape}")

# =============================================================================
# 2. DATA QUALITY VALIDATION
# =============================================================================

def validate_data(df):
    report = {}
    report['missing'] = df.isnull().sum().to_dict()
    report['duplicates'] = df.duplicated().sum()
    report['dtypes'] = df.dtypes.astype(str).to_dict()
    report['outliers'] = {}
    for col in ['math', 'science', 'english', 'history']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))).sum()
        report['outliers'][col] = int(outlier_count)
    print("Data Quality Report:", json.dumps(report, indent=2, default=int))
    return report

quality_report = validate_data(student_df)

# =============================================================================
# 3. MULTI-SUBJECT GRADE ANALYSIS & STATISTICAL SUMMARIES
# =============================================================================

def subject_statistics(df):
    subjects = ['math', 'science', 'english', 'history']
    stats = df[subjects].agg(['mean', 'std', 'min', 'max', 'median', 'count']).T
    stats['pass_rate'] = (df[subjects] >= 40).sum() / len(df)
    print("\nSubject Statistics:\n", stats)
    return stats

subject_stats = subject_statistics(student_df)

# =============================================================================
# 4. TIME-SERIES PERFORMANCE TRACKING
# =============================================================================

def time_series_performance(df):
    # Group by exam_date (monthly)
    df['exam_month'] = df['exam_date'].dt.to_period('M')
    monthly_perf = df.groupby('exam_month')[['math', 'science', 'english', 'history']].mean()
    print("\nMonthly Average Performance:\n", monthly_perf.tail())
    return monthly_perf

monthly_performance = time_series_performance(student_df)

# =============================================================================
# 5. PATTERN & TREND IDENTIFICATION
# =============================================================================

def identify_patterns(df):
    # Correlation between attendance and grades
    subjects = ['math', 'science', 'english', 'history']
    corr = df[subjects + ['attendance_pct']].corr()['attendance_pct'][:-1]
    # Activity impact
    activity_perf = df.groupby('activity')[subjects].mean()
    print("\nAttendance-Grade Correlation:\n", corr)
    print("\nPerformance by Activity:\n", activity_perf)
    return corr, activity_perf

attendance_corr, activity_perf = identify_patterns(student_df)

# =============================================================================
# 6. INSIGHTS & RECOMMENDATIONS
# =============================================================================

def generate_insights(df, subject_stats, attendance_corr):
    insights = []
    for subj in subject_stats.index:
        if subject_stats.loc[subj, 'pass_rate'] < 0.85:
            insights.append(f"Low pass rate in {subj.title()}: {subject_stats.loc[subj, 'pass_rate']:.1%}. Recommend extra classes.")
    for subj, corr in attendance_corr.items():
        if corr > 0.2:
            insights.append(f"Attendance strongly affects {subj.title()} performance (corr={corr:.2f}). Encourage regular attendance.")
    if df['activity'].value_counts().get('None', 0) / len(df) > 0.5:
        insights.append("Over 50% students not in activities. Recommend promoting extracurricular participation.")
    print("\nInsights & Recommendations:")
    for ins in insights:
        print("•", ins)
    return insights

insights = generate_insights(student_df, subject_stats, attendance_corr)

# =============================================================================
# 7. EXPORT RESULTS IN MULTIPLE FORMATS
# =============================================================================

def export_results(df, subject_stats, monthly_perf, insights, outdir="student_analysis_exports"):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "student_data.csv"), index=False)
    subject_stats.to_csv(os.path.join(outdir, "subject_statistics.csv"))
    monthly_perf.to_csv(os.path.join(outdir, "monthly_performance.csv"))
    with open(os.path.join(outdir, "insights.txt"), "w") as f:
        for ins in insights:
            f.write(ins + "\n")
    # Export summary as JSON
    summary = {
        "insights": insights,
        "subject_stats": subject_stats.reset_index().to_dict(orient="records"),
        "monthly_performance": monthly_perf.reset_index().astype(str).to_dict(orient="records")
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExported results to folder: {outdir}")

export_results(student_df, subject_stats, monthly_performance, insights)

print("\n=== Student Performance Analysis Complete ===")