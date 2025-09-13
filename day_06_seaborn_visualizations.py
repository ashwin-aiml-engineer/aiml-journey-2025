import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

print("=== Day 6: Seaborn Advanced Visualizations ===\n")

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. LOADING SAMPLE DATASETS
print("1. Loading Sample Datasets")
print("-" * 30)

# Load built-in seaborn datasets
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')

print("Available datasets loaded:")
print(f"✓ Tips dataset: {tips.shape}")
print(f"✓ Flights dataset: {flights.shape}")
print(f"✓ Iris dataset: {iris.shape}")
print(f"✓ Titanic dataset: {titanic.shape}")

# Display sample data
print("\nTips dataset preview:")
print(tips.head())

# 2. DISTRIBUTION PLOTS
print("\n2. Distribution Plots")
print("-" * 22)

# Create subplot layout for distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution Analysis with Seaborn', fontsize=16, fontweight='bold')

# Histogram with KDE
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram with KDE - Total Bill')

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 1])
axes[0, 1].set_title('Box Plot - Total Bill by Day')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[0, 2])
axes[0, 2].set_title('Violin Plot - Total Bill by Day')

# KDE plot
sns.kdeplot(data=tips, x='total_bill', hue='time', ax=axes[1, 0])
axes[1, 0].set_title('KDE Plot - Total Bill by Time')

# Strip plot
sns.stripplot(data=tips, x='day', y='total_bill', size=4, ax=axes[1, 1])
axes[1, 1].set_title('Strip Plot - Total Bill by Day')

# Swarm plot
sns.swarmplot(data=tips, x='day', y='total_bill', ax=axes[1, 2])
axes[1, 2].set_title('Swarm Plot - Total Bill by Day')

plt.tight_layout()
plt.show()

print("✓ Distribution plots created")

# 3. RELATIONSHIP PLOTS
print("\n3. Relationship Plots")
print("-" * 22)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Relationship Analysis with Seaborn', fontsize=16, fontweight='bold')

# Scatter plot with regression line
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', size='size', ax=axes[0, 0])
axes[0, 0].set_title('Scatter Plot - Tip vs Total Bill')

# Regression plot
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 1])
axes[0, 1].set_title('Regression Plot - Tip vs Total Bill')

# Line plot (using flights data)
flights_pivot = flights.pivot(index='month', columns='year', values='passengers')
sns.lineplot(data=flights, x='month', y='passengers', hue='year', ax=axes[1, 0])
axes[1, 0].set_title('Line Plot - Flight Passengers Over Time')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# Residual plot
sns.residplot(data=tips, x='total_bill', y='tip', ax=axes[1, 1])
axes[1, 1].set_title('Residual Plot - Tip vs Total Bill')

plt.tight_layout()
plt.show()

print("✓ Relationship plots created")

# 4. CATEGORICAL PLOTS
print("\n4. Categorical Plots")
print("-" * 20)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Categorical Data Analysis', fontsize=16, fontweight='bold')

# Count plot
sns.countplot(data=tips, x='day', hue='time', ax=axes[0, 0])
axes[0, 0].set_title('Count Plot - Customers by Day and Time')

# Bar plot
sns.barplot(data=tips, x='day', y='total_bill', hue='time', ax=axes[0, 1])
axes[0, 1].set_title('Bar Plot - Average Bill by Day and Time')

# Point plot
sns.pointplot(data=tips, x='day', y='total_bill', hue='time', ax=axes[0, 2])
axes[0, 2].set_title('Point Plot - Average Bill by Day and Time')

# Box plot with categorical data
sns.boxplot(data=titanic, x='class', y='age', hue='survived', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot - Age by Class and Survival')

# Violin plot with categorical data
sns.violinplot(data=titanic, x='class', y='age', hue='survived', split=True, ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot - Age by Class and Survival')

# Cat plot (factor plot)
sns.catplot(data=tips, x='day', y='total_bill', hue='smoker', kind='box', ax=axes[1, 2])
plt.close()  # Close the extra figure created by catplot
sns.boxplot(data=tips, x='day', y='total_bill', hue='smoker', ax=axes[1, 2])
axes[1, 2].set_title('Categorical Plot - Bill by Day and Smoking')

plt.tight_layout()
plt.show()

print("✓ Categorical plots created")

# 5. MATRIX PLOTS (HEATMAPS)
print("\n5. Matrix Plots (Heatmaps)")
print("-" * 28)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Matrix Visualizations - Correlation and Patterns', fontsize=16, fontweight='bold')

# Correlation heatmap
numeric_tips = tips.select_dtypes(include=[np.number])
correlation_matrix = numeric_tips.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[0])
axes[0].set_title('Correlation Heatmap - Tips Dataset')

# Pivot table heatmap
flights_pivot = flights.pivot(index='month', columns='year', values='passengers')
sns.heatmap(flights_pivot, annot=True, cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Heatmap - Flight Passengers by Month/Year')

# Iris correlation heatmap
iris_numeric = iris.select_dtypes(include=[np.number])
iris_corr = iris_numeric.corr()
sns.heatmap(iris_corr, annot=True, cmap='viridis', square=True, ax=axes[2])
axes[2].set_title('Correlation Heatmap - Iris Dataset')

plt.tight_layout()
plt.show()

print("✓ Matrix plots created")

# 6. MULTI-PLOT GRIDS
print("\n6. Multi-plot Grids")
print("-" * 20)

# Pair plot for exploring relationships
print("Creating pair plot...")
pair_plot = sns.pairplot(iris, hue='species', markers=['o', 's', 'D'])
pair_plot.fig.suptitle('Pair Plot - Iris Dataset Relationships', y=1.02, fontsize=16)
plt.show()

# FacetGrid for custom multi-plot layouts
print("Creating FacetGrid...")
g = sns.FacetGrid(tips, col='time', row='smoker', margin_titles=True, height=4)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.7)
g.add_legend()
g.fig.suptitle('FacetGrid - Tips Analysis by Time and Smoking', y=1.02, fontsize=16)
plt.show()

print("✓ Multi-plot grids created")

# 7. ADVANCED CUSTOMIZATION
print("\n7. Advanced Customization")
print("-" * 27)

# Create a publication-ready plot
plt.figure(figsize=(14, 8))

# Custom color palette
custom_palette = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
sns.set_palette(custom_palette)

# Complex plot with multiple elements
ax = sns.boxplot(data=tips, x='day', y='total_bill', hue='time')
sns.stripplot(data=tips, x='day', y='total_bill', hue='time', 
              dodge=True, alpha=0.6, size=3)

# Customization
plt.title('Restaurant Tips Analysis\nTotal Bill Distribution by Day and Dining Time', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
plt.ylabel('Total Bill ($)', fontsize=12, fontweight='bold')

# Handle legends
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[2:4], labels[2:4], title='Dining Time', 
           title_fontsize=12, fontsize=10)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Advanced customized plot created")

# 8. STATISTICAL ANALYSIS PLOTS
print("\n8. Statistical Analysis Plots")
print("-" * 31)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Statistical Analysis Visualizations', fontsize=16, fontweight='bold')

# Joint plot data preparation
np.random.seed(42)
x_data = np.random.normal(0, 1, 1000)
y_data = 2 * x_data + np.random.normal(0, 1, 1000)

# Regression plot with confidence interval
sns.regplot(x=x_data, y=y_data, ax=axes[0, 0])
axes[0, 0].set_title('Regression with Confidence Interval')

# Distribution comparison
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(105, 12, 1000)
sns.histplot(data1, alpha=0.5, label='Group A', ax=axes[0, 1])
sns.histplot(data2, alpha=0.5, label='Group B', ax=axes[0, 1])
axes[0, 1].set_title('Distribution Comparison')
axes[0, 1].legend()

# Box plot comparison
comparison_data = pd.DataFrame({
    'Group A': data1[:500],
    'Group B': data2[:500],
    'Group C': np.random.normal(95, 18, 500)
})
sns.boxplot(data=comparison_data, ax=axes[1, 0])
axes[1, 0].set_title('Statistical Summary Comparison')

# Violin plot with quartiles
sns.violinplot(data=comparison_data, ax=axes[1, 1], inner='quartile')
axes[1, 1].set_title('Distribution Shape Comparison')

plt.tight_layout()
plt.show()

print("✓ Statistical analysis plots created")

# 9. BEST PRACTICES AND TIPS
print("\n9. Seaborn Best Practices")
print("-" * 27)

best_practices = [
    "Choose appropriate plot types based on data types and relationships",
    "Use color meaningfully - categorical vs continuous variables",
    "Always include proper titles, axis labels, and legends",
    "Consider your audience when choosing complexity level",
    "Use subplots for comparing multiple relationships",
    "Leverage built-in statistical functionality (confidence intervals, regression)",
    "Customize color palettes for brand consistency",
    "Save plots in appropriate formats and resolutions"
]

for i, practice in enumerate(best_practices, 1):
    print(f"{i}. {practice}")

# 10. SUMMARY STATISTICS
print("\n10. Data Summary")
print("-" * 17)

print("Tips Dataset Summary Statistics:")
print(tips.describe())

print("\nIris Dataset Summary by Species:")
print(iris.groupby('species').describe())

print("\n" + "="*50)
print("DAY 6 VISUALIZATION MASTERY COMPLETED! 🎨📊")
print("="*50)
print("\nKey Achievements:")
print("✓ Mastered matplotlib fundamentals")
print("✓ Advanced seaborn statistical visualizations")
print("✓ Created publication-ready plots")
print("✓ Learned data exploration through visualization")
print("✓ Applied statistical plotting techniques")
print("✓ Built comprehensive visualization workflows")
print("\nVisualization Skills Unlocked:")
print("• Distribution analysis")
print("• Relationship exploration")  
print("• Categorical data visualization")
print("• Correlation and pattern recognition")
print("• Multi-dimensional data exploration")

