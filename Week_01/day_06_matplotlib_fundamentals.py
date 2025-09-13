import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=== Day 6: Matplotlib Fundamentals ===\n")

# 1. BASIC PLOTTING CONCEPTS
print("1. Basic Plotting Concepts")
print("-" * 30)

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Basic Line Plot - Sine Wave')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Multiple lines on same plot
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='blue')
plt.plot(x, np.cos(x), label='cos(x)', color='red')
plt.plot(x, np.tan(x), label='tan(x)', color='green', alpha=0.7)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Multiple Functions Comparison')
plt.legend()
plt.ylim(-2, 2)  # Limit y-axis for better visualization
plt.grid(True, alpha=0.3)
plt.show()

print("✓ Basic line plots created")

# 2. SCATTER PLOTS
print("\n2. Scatter Plots")
print("-" * 20)

# Generate sample data
np.random.seed(42)
n_points = 100
x_scatter = np.random.normal(0, 1, n_points)
y_scatter = 2 * x_scatter + np.random.normal(0, 0.5, n_points)
colors = np.random.rand(n_points)
sizes = 1000 * np.random.rand(n_points)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color Scale')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Variable Colors and Sizes')
plt.grid(True, alpha=0.3)
plt.show()

print("✓ Scatter plot with customizations created")

# 3. BAR CHARTS
print("\n3. Bar Charts")
print("-" * 15)

# Simple bar chart
categories = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust']
popularity = [85, 75, 70, 60, 45, 35]

plt.figure(figsize=(12, 6))

# Vertical bar chart
plt.subplot(1, 2, 1)
bars = plt.bar(categories, popularity, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
plt.xlabel('Programming Languages')
plt.ylabel('Popularity Score')
plt.title('Programming Language Popularity (Vertical)')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, value in zip(bars, popularity):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(value), ha='center', va='bottom')

# Horizontal bar chart
plt.subplot(1, 2, 2)
plt.barh(categories, popularity, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
plt.xlabel('Popularity Score')
plt.ylabel('Programming Languages')
plt.title('Programming Language Popularity (Horizontal)')

plt.tight_layout()
plt.show()

print("✓ Bar charts (vertical and horizontal) created")

# 4. HISTOGRAMS
print("\n4. Histograms")
print("-" * 15)

# Generate sample data
np.random.seed(42)
data1 = np.random.normal(100, 15, 1000)  # Normal distribution
data2 = np.random.exponential(2, 1000)   # Exponential distribution
data3 = np.random.uniform(0, 10, 1000)   # Uniform distribution

plt.figure(figsize=(15, 5))

# Normal distribution histogram
plt.subplot(1, 3, 1)
plt.hist(data1, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Normal Distribution')
plt.axvline(np.mean(data1), color='red', linestyle='--', label=f'Mean: {np.mean(data1):.1f}')
plt.legend()

# Exponential distribution histogram
plt.subplot(1, 3, 2)
plt.hist(data2, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Exponential Distribution')
plt.axvline(np.mean(data2), color='red', linestyle='--', label=f'Mean: {np.mean(data2):.1f}')
plt.legend()

# Uniform distribution histogram
plt.subplot(1, 3, 3)
plt.hist(data3, bins=30, color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Uniform Distribution')
plt.axvline(np.mean(data3), color='red', linestyle='--', label=f'Mean: {np.mean(data3):.1f}')
plt.legend()

plt.tight_layout()
plt.show()

print("✓ Histograms for different distributions created")

# 5. SUBPLOTS AND COMPLEX LAYOUTS
print("\n5. Subplots and Complex Layouts")
print("-" * 35)

# Create complex subplot layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Complex Subplot Layout - Data Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'b-', label='sin(x)')
axes[0, 0].plot(x, np.cos(x), 'r--', label='cos(x)')
axes[0, 0].set_title('Trigonometric Functions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100)
axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, color='purple')
axes[0, 1].set_title('Random Correlation')
axes[0, 1].set_xlabel('X values')
axes[0, 1].set_ylabel('Y values')

# Plot 3: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[0, 2].bar(categories, values, color='orange', alpha=0.7)
axes[0, 2].set_title('Category Analysis')
axes[0, 2].set_ylabel('Values')

# Plot 4: Histogram
data = np.random.normal(50, 15, 1000)
axes[1, 0].hist(data, bins=25, color='lightblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution Analysis')
axes[1, 0].set_xlabel('Values')
axes[1, 0].set_ylabel('Frequency')

# Plot 5: Pie chart
sizes = [35, 25, 20, 15, 5]
labels = ['Product A', 'Product B', 'Product C', 'Product D', 'Others']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Market Share Distribution')

# Plot 6: Box plot
data_box = [np.random.normal(0, 1, 100), np.random.normal(1, 1.5, 100), 
            np.random.normal(-1, 0.5, 100), np.random.normal(2, 2, 100)]
axes[1, 2].boxplot(data_box, labels=['Group A', 'Group B', 'Group C', 'Group D'])
axes[1, 2].set_title('Group Comparison')
axes[1, 2].set_ylabel('Values')

plt.tight_layout()
plt.show()

print("✓ Complex subplot layout created")

# 6. PLOT CUSTOMIZATION AND STYLING
print("\n6. Plot Customization and Styling")
print("-" * 38)

# Create a professional-looking plot
plt.style.use('seaborn-v0_8')  # Use seaborn style
fig, ax = plt.subplots(figsize=(12, 8))

# Generate time series data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
price_data = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
volume_data = np.random.exponential(1000, len(dates))

# Main price plot
color_main = '#2E86AB'
ax.plot(dates, price_data, color=color_main, linewidth=2.5, label='Stock Price')
ax.fill_between(dates, price_data, alpha=0.3, color=color_main)

# Customization
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color=color_main)
ax.set_title('Stock Price Analysis - 2024\nProfessional Visualization Example', 
             fontsize=16, fontweight='bold', pad=20)

# Add annotations
max_price_idx = np.argmax(price_data)
ax.annotate(f'Peak: ${price_data[max_price_idx]:.2f}', 
            xy=(dates[max_price_idx], price_data[max_price_idx]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Secondary y-axis for volume
ax2 = ax.twinx()
ax2.bar(dates, volume_data, alpha=0.3, color='orange', width=1, label='Volume')
ax2.set_ylabel('Volume', fontsize=12, fontweight='bold', color='orange')

# Legends
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Professional styled plot with dual y-axis created")

# 7. SAVING PLOTS
print("\n7. Saving Plots")
print("-" * 17)

# Create a plot to save
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='Sine Wave', linewidth=3, color='#E74C3C')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Example Plot for Saving')
ax.legend()
ax.grid(True, alpha=0.3)

# Save in different formats
formats = ['png', 'pdf', 'svg']
for fmt in formats:
    filename = f'day_06_example_plot.{fmt}'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Plot saved as {filename}")

plt.show()

# 8. MATPLOTLIB BEST PRACTICES SUMMARY
print("\n8. Matplotlib Best Practices")
print("-" * 32)

best_practices = [
    "Always use descriptive titles and axis labels",
    "Include legends when plotting multiple series",
    "Use appropriate color schemes and transparency",
    "Save plots in high resolution (300+ DPI) for presentations",
    "Use grid for better readability",
    "Choose appropriate plot types for your data",
    "Keep plots clean and avoid clutter",
    "Use consistent styling across related plots"
]

for i, practice in enumerate(best_practices, 1):
    print(f"{i}. {practice}")

print("\n" + "="*50)
print("DAY 6 COMPLETED SUCCESSFULLY! 🎉")
print("="*50)
print("\nKey Achievements:")
print("✓ Mastered basic matplotlib plotting")
print("✓ Created various plot types (line, scatter, bar, histogram)")
print("✓ Learned subplot layouts and complex visualizations")
print("✓ Applied professional styling and customization")
print("✓ Understood plot saving and export options")

