import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 Day 6 Mini Project: Sales Analytics Dashboard")
print("=" * 55)

# 1. DATA GENERATION
print("\n1. Generating Synthetic Sales Data...")
print("-" * 40)

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Product categories and details
products = {
    'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera'],
    'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket'],
    'Home & Garden': ['Furniture', 'Kitchen Appliances', 'Bedding', 'Plants', 'Tools'],
    'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Biography'],
    'Sports': ['Equipment', 'Apparel', 'Accessories', 'Supplements', 'Footwear']
}

# Sales representatives
sales_reps = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown',
              'Frank Miller', 'Grace Lee', 'Henry Taylor', 'Iris Chen', 'Jack Anderson']

# Regions
regions = ['North', 'South', 'East', 'West', 'Central']

# Generate comprehensive sales data
n_records = 5000
sales_data = []

for i in range(n_records):
    # Random date with seasonal patterns
    date = np.random.choice(date_range)
    date = pd.Timestamp(date)  # Ensure it's a pandas Timestamp
    month = date.month
    
    # Seasonal multiplier (higher sales in Q4)
    seasonal_multiplier = 1.0
    if month in [11, 12]:  # November, December
        seasonal_multiplier = 1.8
    elif month in [6, 7, 8]:  # Summer months
        seasonal_multiplier = 1.3
    
    # Select product category and product
    category = np.random.choice(list(products.keys()))
    product = np.random.choice(products[category])
    
    # Generate sales amount based on category
    base_amounts = {
        'Electronics': (200, 2000),
        'Clothing': (25, 300),
        'Home & Garden': (50, 1500),
        'Books': (10, 100),
        'Sports': (30, 500)
    }
    
    min_amount, max_amount = base_amounts[category]
    sales_amount = np.random.uniform(min_amount, max_amount) * seasonal_multiplier
    
    # Generate quantity
    if category == 'Electronics':
        quantity = np.random.randint(1, 5)
    elif category == 'Books':
        quantity = np.random.randint(1, 10)
    else:
        quantity = np.random.randint(1, 8)
    
    # Add some correlation between sales amount and quantity
    if quantity > 3:
        sales_amount *= 1.2
    
    # Customer satisfaction (correlated with sales rep performance)
    sales_rep = np.random.choice(sales_reps)
    base_satisfaction = np.random.normal(4.2, 0.8)
    if sales_rep in ['Alice Johnson', 'Carol Davis', 'Eva Brown']:  # Top performers
        base_satisfaction += 0.5
    
    customer_satisfaction = np.clip(base_satisfaction, 1, 5)
    
    sales_data.append({
        'Date': date,
        'Category': category,
        'Product': product,
        'Sales_Amount': round(sales_amount, 2),
        'Quantity': quantity,
        'Sales_Rep': sales_rep,
        'Region': np.random.choice(regions),
        'Customer_Satisfaction': round(customer_satisfaction, 1)
    })

# Create DataFrame
df = pd.DataFrame(sales_data)

# Add derived columns
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year
df['Day_of_Week'] = df['Date'].dt.day_name()
df['Revenue_per_Unit'] = df['Sales_Amount'] / df['Quantity']

print(f"✓ Generated {len(df)} sales records")
print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"✓ Categories: {', '.join(df['Category'].unique())}")
print(f"✓ Total revenue: ${df['Sales_Amount'].sum():,.2f}")

# Display sample data
print("\nSample Data:")
print(df.head())

# 2. DATA ANALYSIS SUMMARY
print("\n2. Data Analysis Summary")
print("-" * 28)

print("Dataset Overview:")
print(f"• Total Records: {len(df):,}")
print(f"• Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"• Total Revenue: ${df['Sales_Amount'].sum():,.2f}")
print(f"• Average Order Value: ${df['Sales_Amount'].mean():.2f}")
print(f"• Total Units Sold: {df['Quantity'].sum():,}")

print("\nTop Performing Categories:")
category_performance = df.groupby('Category').agg({
    'Sales_Amount': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales_Amount', ascending=False)
print(category_performance)

# 3. COMPREHENSIVE DASHBOARD CREATION
print("\n3. Creating Sales Analytics Dashboard...")
print("-" * 42)

# Set up the dashboard layout
plt.style.use('default')
sns.set_palette("husl")
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Sales Analytics Dashboard 2023-2024\nComprehensive Business Intelligence Report', 
             fontsize=20, fontweight='bold', y=0.98)

# Define grid layout (6 rows, 3 columns)
gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)

# 1. Revenue by Category (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
category_revenue = df.groupby('Category')['Sales_Amount'].sum().sort_values(ascending=True)
bars = ax1.barh(category_revenue.index, category_revenue.values, 
                color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'])
ax1.set_title('Total Revenue by Category', fontweight='bold', fontsize=12)
ax1.set_xlabel('Revenue ($)')
for i, v in enumerate(category_revenue.values):
    ax1.text(v + max(category_revenue.values) * 0.01, i, f'${v:,.0f}', 
             va='center', fontweight='bold')

# 2. Monthly Sales Trend (Top Middle)
ax2 = fig.add_subplot(gs[0, 1])
monthly_sales = df.groupby(['Year', 'Month'])['Sales_Amount'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
ax2.plot(monthly_sales['Date'], monthly_sales['Sales_Amount'], 
         marker='o', linewidth=2, markersize=6, color='#E74C3C')
ax2.fill_between(monthly_sales['Date'], monthly_sales['Sales_Amount'], alpha=0.3, color='#E74C3C')
ax2.set_title('Monthly Sales Trend', fontweight='bold', fontsize=12)
ax2.set_ylabel('Revenue ($)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# 3. Sales Rep Performance (Top Right)
ax3 = fig.add_subplot(gs[0, 2])
rep_performance = df.groupby('Sales_Rep').agg({
    'Sales_Amount': 'sum',
    'Customer_Satisfaction': 'mean'
}).sort_values('Sales_Amount', ascending=False)
scatter = ax3.scatter(rep_performance['Sales_Amount'], rep_performance['Customer_Satisfaction'],
                     s=rep_performance['Sales_Amount']/500, alpha=0.6, c=range(len(rep_performance)),
                     cmap='viridis')
ax3.set_xlabel('Total Sales ($)')
ax3.set_ylabel('Avg Customer Satisfaction')
ax3.set_title('Sales Rep Performance\n(Bubble size = Total Sales)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# 4. Regional Analysis (Middle Left)
ax4 = fig.add_subplot(gs[1, 0])
regional_data = df.groupby('Region')['Sales_Amount'].sum()
wedges, texts, autotexts = ax4.pie(regional_data.values, labels=regional_data.index, 
                                  autopct='%1.1f%%', startangle=90,
                                  colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
ax4.set_title('Revenue Distribution by Region', fontweight='bold', fontsize=12)

# 5. Quarterly Growth (Middle Middle)
ax5 = fig.add_subplot(gs[1, 1])
quarterly_data = df.groupby(['Year', 'Quarter'])['Sales_Amount'].sum().reset_index()
quarterly_data['Period'] = quarterly_data['Year'].astype(str) + '-Q' + quarterly_data['Quarter'].astype(str)
bars = ax5.bar(quarterly_data['Period'], quarterly_data['Sales_Amount'],
               color=['#3498DB' if '2023' in period else '#E74C3C' for period in quarterly_data['Period']])
ax5.set_title('Quarterly Sales Performance', fontweight='bold', fontsize=12)
ax5.set_ylabel('Revenue ($)')
ax5.tick_params(axis='x', rotation=45)
for bar, value in zip(bars, quarterly_data['Sales_Amount']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quarterly_data['Sales_Amount'])*0.01,
             f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

# 6. Product Performance Heatmap (Middle Right)
ax6 = fig.add_subplot(gs[1, 2])
product_month = df.groupby(['Category', 'Month'])['Sales_Amount'].sum().unstack(fill_value=0)
sns.heatmap(product_month, annot=False, cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Revenue ($)'})
ax6.set_title('Category Performance by Month', fontweight='bold', fontsize=12)
ax6.set_xlabel('Month')
ax6.set_ylabel('Category')

# 7. Daily Sales Pattern (Bottom Left)
ax7 = fig.add_subplot(gs[2, 0])
daily_pattern = df.groupby('Day_of_Week')['Sales_Amount'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
bars = ax7.bar(range(len(daily_pattern)), daily_pattern.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
ax7.set_title('Average Daily Sales Pattern', fontweight='bold', fontsize=12)
ax7.set_ylabel('Average Revenue ($)')
ax7.set_xticks(range(len(daily_pattern)))
ax7.set_xticklabels([day[:3] for day in daily_pattern.index], rotation=45)

# 8. Customer Satisfaction Distribution (Bottom Middle)
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(df['Customer_Satisfaction'], bins=20, color='#3498DB', alpha=0.7, edgecolor='black')
ax8.axvline(df['Customer_Satisfaction'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["Customer_Satisfaction"].mean():.2f}')
ax8.set_title('Customer Satisfaction Distribution', fontweight='bold', fontsize=12)
ax8.set_xlabel('Satisfaction Rating')
ax8.set_ylabel('Frequency')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Revenue vs Quantity Analysis (Bottom Right)
ax9 = fig.add_subplot(gs[2, 2])
for category in df['Category'].unique():
    cat_data = df[df['Category'] == category]
    ax9.scatter(cat_data['Quantity'], cat_data['Sales_Amount'], 
               label=category, alpha=0.6, s=30)
ax9.set_xlabel('Quantity Sold')
ax9.set_ylabel('Sales Amount ($)')
ax9.set_title('Revenue vs Quantity by Category', fontweight='bold', fontsize=12)
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax9.grid(True, alpha=0.3)

# 10. Sales Rep Detailed Analysis (Row 4, Spans 2 columns)
ax10 = fig.add_subplot(gs[3, :2])
rep_detailed = df.groupby('Sales_Rep').agg({
    'Sales_Amount': ['sum', 'mean'],
    'Quantity': 'sum',
    'Customer_Satisfaction': 'mean'
}).round(2)
rep_detailed.columns = ['Total_Sales', 'Avg_Sale', 'Total_Quantity', 'Avg_Satisfaction']
rep_detailed = rep_detailed.sort_values('Total_Sales', ascending=True)

# Create a horizontal bar chart
y_pos = np.arange(len(rep_detailed))
bars = ax10.barh(y_pos, rep_detailed['Total_Sales'], color='lightblue', alpha=0.7)
ax10.set_yticks(y_pos)
ax10.set_yticklabels(rep_detailed.index)
ax10.set_xlabel('Total Sales ($)')
ax10.set_title('Sales Representative Performance Ranking', fontweight='bold', fontsize=12)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, rep_detailed['Total_Sales'])):
    ax10.text(bar.get_width() + max(rep_detailed['Total_Sales']) * 0.01, 
              bar.get_y() + bar.get_height()/2, f'${value:,.0f}', 
              va='center', fontweight='bold')

# 11. Customer Satisfaction by Category (Row 4, Right)
ax11 = fig.add_subplot(gs[3, 2])
sns.boxplot(data=df, x='Customer_Satisfaction', y='Category', ax=ax11)
ax11.set_title('Customer Satisfaction by Category', fontweight='bold', fontsize=12)
ax11.set_xlabel('Customer Satisfaction Rating')

# 12. Time Series with Moving Average (Row 5, Full Width)
ax12 = fig.add_subplot(gs[4, :])
daily_sales = df.groupby('Date')['Sales_Amount'].sum().reset_index()
daily_sales = daily_sales.sort_values('Date')
daily_sales['MA_7'] = daily_sales['Sales_Amount'].rolling(window=7).mean()
daily_sales['MA_30'] = daily_sales['Sales_Amount'].rolling(window=30).mean()

ax12.plot(daily_sales['Date'], daily_sales['Sales_Amount'], alpha=0.3, color='gray', label='Daily Sales')
ax12.plot(daily_sales['Date'], daily_sales['MA_7'], color='blue', linewidth=2, label='7-Day Moving Average')
ax12.plot(daily_sales['Date'], daily_sales['MA_30'], color='red', linewidth=2, label='30-Day Moving Average')
ax12.set_title('Sales Trend Analysis with Moving Averages', fontweight='bold', fontsize=14)
ax12.set_ylabel('Sales Amount ($)')
ax12.legend()
ax12.grid(True, alpha=0.3)

# 13. Key Performance Indicators (Row 6, Full Width)
ax13 = fig.add_subplot(gs[5, :])
ax13.axis('off')

# Calculate KPIs
total_revenue = df['Sales_Amount'].sum()
total_orders = len(df)
avg_order_value = df['Sales_Amount'].mean()
total_customers = df['Sales_Rep'].nunique() * 50  # Estimate
avg_satisfaction = df['Customer_Satisfaction'].mean()
top_category = df.groupby('Category')['Sales_Amount'].sum().idxmax()
growth_rate = ((df[df['Year']==2024]['Sales_Amount'].sum() / 
                df[df['Year']==2023]['Sales_Amount'].sum()) - 1) * 100

# Create KPI boxes
kpis = [
    ('Total Revenue', f'${total_revenue:,.0f}', '#E74C3C'),
    ('Total Orders', f'{total_orders:,}', '#3498DB'),
    ('Avg Order Value', f'${avg_order_value:.2f}', '#2ECC71'),
    ('Estimated Customers', f'{total_customers:,}', '#F39C12'),
    ('Avg Satisfaction', f'{avg_satisfaction:.2f}/5.0', '#9B59B6'),
    ('Top Category', top_category, '#1ABC9C'),
    ('YoY Growth', f'{growth_rate:.1f}%', '#E67E22')
]

kpi_width = 1.0 / len(kpis)
for i, (title, value, color) in enumerate(kpis):
    x_pos = i * kpi_width + kpi_width/2
    
    # Create colored box
    box = plt.Rectangle((i * kpi_width + 0.01, 0.3), kpi_width - 0.02, 0.4, 
                       facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax13.add_patch(box)
    
    # Add text
    ax13.text(x_pos, 0.6, title, ha='center', va='center', fontweight='bold', fontsize=10)
    ax13.text(x_pos, 0.4, value, ha='center', va='center', fontweight='bold', fontsize=12, color=color)

ax13.set_xlim(0, 1)
ax13.set_ylim(0, 1)
ax13.set_title('Key Performance Indicators', fontweight='bold', fontsize=14, y=0.9)

plt.tight_layout()
plt.savefig('sales_analytics_dashboard.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("✓ Comprehensive dashboard created and saved!")

# 4. STATISTICAL INSIGHTS
print("\n4. Key Business Insights")
print("-" * 26)

insights = [
    f"📊 Total Revenue: ${total_revenue:,.2f} across {total_orders:,} transactions",
    f"📈 Year-over-Year Growth: {growth_rate:.1f}%",
    f"🏆 Top Category: {top_category} (${df.groupby('Category')['Sales_Amount'].sum()[top_category]:,.0f})",
    f"⭐ Average Customer Satisfaction: {avg_satisfaction:.2f}/5.0",
    f"💰 Average Order Value: ${avg_order_value:.2f}",
    f"🎯 Best Sales Rep: {rep_performance.index[0]} (${rep_performance.iloc[0]['Sales_Amount']:,.0f})",
    f"📅 Peak Sales Day: {daily_pattern.idxmax()} (${daily_pattern.max():,.0f} avg)",
    f"🚀 Best Quarter: Q4 2024 with highest seasonal performance"
]

for insight in insights:
    print(f"  {insight}")

# 5. EXPORT SUMMARY
print("\n5. Export Summary")
print("-" * 18)

# Save data summary
summary_stats = df.describe()
summary_stats.to_csv('sales_data_summary.csv')

# Save processed data
df.to_csv('complete_sales_data.csv', index=False)

print("✓ Dashboard saved as: sales_analytics_dashboard.png")
print("✓ Data summary saved as: sales_data_summary.csv") 
print("✓ Complete dataset saved as: complete_sales_data.csv")