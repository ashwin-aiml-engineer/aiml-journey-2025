"""
Day 7: Complete Data Science Portfolio Project
AI/ML Engineer 90-Day Challenge - Week 1 Capstone

Project: E-Commerce Customer Analytics & Business Intelligence Platform
=========================================================================

Business Problem:
An e-commerce company wants to understand their customer behavior, 
identify trends, and optimize their business strategy using data-driven insights.

Skills Demonstrated:
- Python programming (OOP, functions, error handling)
- Data structures and algorithms
- NumPy and Pandas for data manipulation
- Statistical analysis and insights
- Professional data visualization
- Business intelligence dashboard creation
- End-to-end data pipeline development

Project Structure:
1. Data Generation & Simulation
2. Data Processing & Analysis Class
3. Statistical Analysis & Insights
4. Comprehensive Visualization Dashboard  
5. Business Intelligence Report
6. Export & Documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

print("🚀 E-COMMERCE CUSTOMER ANALYTICS PLATFORM")
print("=" * 55)
print("Day 7: Week 1 Portfolio Project")
print("Complete Data Science Pipeline Demonstration")
print("=" * 55)

# ================================
# 1. DATA GENERATION ENGINE
# ================================

class ECommerceDataGenerator:
    """
    Advanced data generation engine for realistic e-commerce data
    Demonstrates OOP principles and complex data relationships
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with reproducible randomness"""
        np.random.seed(seed)
        self.products = self._initialize_products()
        self.customers = self._initialize_customers()
        self.generated_data = None
        
    def _initialize_products(self):
        """Create comprehensive product catalog"""
        return {
            'Electronics': {
                'items': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera', 'Smart Watch'],
                'price_range': (50, 2000),
                'profit_margin': 0.25,
                'seasonality': {'Q1': 1.0, 'Q2': 1.1, 'Q3': 1.2, 'Q4': 1.8}
            },
            'Clothing': {
                'items': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Accessories'],
                'price_range': (15, 300), 
                'profit_margin': 0.45,
                'seasonality': {'Q1': 0.8, 'Q2': 1.2, 'Q3': 1.0, 'Q4': 1.6}
            },
            'Home & Garden': {
                'items': ['Furniture', 'Kitchen Appliances', 'Bedding', 'Plants', 'Tools', 'Decor'],
                'price_range': (20, 1500),
                'profit_margin': 0.35,
                'seasonality': {'Q1': 1.1, 'Q2': 1.4, 'Q3': 1.1, 'Q4': 1.2}
            },
            'Books': {
                'items': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Biography', 'Tech'],
                'price_range': (10, 80),
                'profit_margin': 0.40,
                'seasonality': {'Q1': 1.2, 'Q2': 0.9, 'Q3': 1.3, 'Q4': 1.1}
            },
            'Sports': {
                'items': ['Equipment', 'Apparel', 'Accessories', 'Supplements', 'Footwear', 'Outdoor'],
                'price_range': (25, 500),
                'profit_margin': 0.30,
                'seasonality': {'Q1': 0.9, 'Q2': 1.3, 'Q3': 1.4, 'Q4': 0.8}
            }
        }
    
    def _initialize_customers(self):
        """Create customer segments with different behaviors"""
        return {
            'Premium': {
                'ratio': 0.15,
                'avg_order_multiplier': 2.5,
                'frequency_multiplier': 2.0,
                'satisfaction_boost': 0.8
            },
            'Regular': {
                'ratio': 0.60,
                'avg_order_multiplier': 1.0,
                'frequency_multiplier': 1.0,
                'satisfaction_boost': 0.0
            },
            'Budget': {
                'ratio': 0.25,
                'avg_order_multiplier': 0.6,
                'frequency_multiplier': 0.7,
                'satisfaction_boost': -0.2
            }
        }
    
    def generate_transactions(self, n_records=10000, date_range_days=730):
        """Generate comprehensive transaction dataset"""
        print(f"\n🔧 Generating {n_records:,} transaction records...")
        
        # Date range setup
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range_days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        transactions = []
        customer_id_counter = 1
        
        for i in range(n_records):
            # Progress indicator
            if i % 1000 == 0:
                print(f"   Progress: {i/n_records*100:.1f}% ({i:,}/{n_records:,})")
            
            # Generate transaction date
            transaction_date = np.random.choice(date_range)
            transaction_date = pd.Timestamp(transaction_date)  # <-- Fix: convert to Timestamp
            quarter = f"Q{(transaction_date.month - 1) // 3 + 1}"
            
            # Select customer segment
            segment_choice = np.random.choice(
                list(self.customers.keys()),
                p=[self.customers[seg]['ratio'] for seg in self.customers.keys()]
            )
            customer_segment = self.customers[segment_choice]
            
            # Select product category and item
            category = np.random.choice(list(self.products.keys()))
            product_info = self.products[category]
            product_name = np.random.choice(product_info['items'])
            
            # Calculate base price with seasonality
            min_price, max_price = product_info['price_range']
            base_price = np.random.uniform(min_price, max_price)
            seasonal_price = base_price * product_info['seasonality'][quarter]
            
            # Apply customer segment pricing
            final_price = seasonal_price * customer_segment['avg_order_multiplier']
            
            # Generate quantity (customer behavior dependent)
            base_quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.4, 0.25, 0.2, 0.1, 0.03, 0.02])
            quantity = max(1, int(base_quantity * customer_segment['frequency_multiplier']))
            
            # Calculate metrics
            revenue = final_price * quantity
            profit = revenue * product_info['profit_margin']
            
            # Customer satisfaction (influenced by multiple factors)
            base_satisfaction = np.random.normal(4.0, 0.8)
            satisfaction_factors = (
                customer_segment['satisfaction_boost'] +
                (0.3 if final_price > 100 else 0) +  # Higher price = higher expectation
                np.random.normal(0, 0.3)  # Random variation
            )
            customer_satisfaction = np.clip(base_satisfaction + satisfaction_factors, 1, 5)
            
            # Geographic and channel data
            regions = ['North', 'South', 'East', 'West', 'Central']
            channels = ['Website', 'Mobile App', 'Physical Store', 'Third Party']
            region = np.random.choice(regions)
            channel = np.random.choice(channels, p=[0.4, 0.35, 0.15, 0.1])
            
            # Create transaction record
            transaction = {
                'transaction_id': f'TXN_{i+1:06d}',
                'customer_id': f'CUST_{customer_id_counter:05d}',
                'customer_segment': segment_choice,
                'transaction_date': transaction_date,
                'category': category,
                'product_name': product_name,
                'quantity': quantity,
                'unit_price': round(final_price, 2),
                'total_revenue': round(revenue, 2),
                'profit': round(profit, 2),
                'customer_satisfaction': round(customer_satisfaction, 1),
                'region': region,
                'channel': channel,
                'quarter': quarter,
                'month': transaction_date.month,
                'year': transaction_date.year,
                'day_of_week': transaction_date.strftime('%A')
            }
            
            transactions.append(transaction)
            
            # Vary customer ID (some customers make multiple purchases)
            if np.random.random() > 0.7:  # 30% chance of new customer
                customer_id_counter += 1
        
        # Create DataFrame
        self.generated_data = pd.DataFrame(transactions)
        
        # Add derived metrics
        self._add_derived_metrics()
        
        print(f"✅ Successfully generated {len(self.generated_data):,} transactions")
        return self.generated_data
    
    def _add_derived_metrics(self):
        """Add calculated fields for analysis"""
        df = self.generated_data
        
        # Profit margin percentage
        df['profit_margin_pct'] = (df['profit'] / df['total_revenue']) * 100
        
        # Revenue per unit
        df['revenue_per_unit'] = df['total_revenue'] / df['quantity']
        
        # High value transaction flag
        df['high_value'] = df['total_revenue'] > df['total_revenue'].quantile(0.8)
        
        # Customer lifetime value approximation
        customer_totals = df.groupby('customer_id')['total_revenue'].sum()
        df['customer_ltv'] = df['customer_id'].map(customer_totals)
        
        # Seasonal indicators
        df['is_holiday_quarter'] = df['quarter'].isin(['Q4'])
        df['is_summer'] = df['quarter'].isin(['Q2', 'Q3'])
        
        print("✅ Added derived metrics for advanced analysis")

# ================================
# 2. DATA ANALYSIS ENGINE
# ================================

class ECommerceAnalyzer:
    """
    Comprehensive data analysis engine
    Demonstrates advanced Pandas operations and statistical analysis
    """
    
    def __init__(self, data):
        """Initialize analyzer with transaction data"""
        self.data = data.copy()
        self.insights = {}
        self.summary_stats = None
        
    def perform_comprehensive_analysis(self):
        """Execute full analysis pipeline"""
        print("\n📊 PERFORMING COMPREHENSIVE DATA ANALYSIS")
        print("-" * 50)
        
        self._basic_statistics()
        self._customer_analysis()
        self._product_analysis()
        self._temporal_analysis()
        self._regional_analysis()
        self._advanced_insights()
        
        return self.insights
    
    def _basic_statistics(self):
        """Generate fundamental dataset statistics"""
        print("🔍 Basic Dataset Statistics...")
        
        df = self.data
        
        basic_stats = {
            'total_transactions': len(df),
            'total_revenue': df['total_revenue'].sum(),
            'total_profit': df['profit'].sum(),
            'avg_order_value': df['total_revenue'].mean(),
            'avg_profit_margin': df['profit_margin_pct'].mean(),
            'unique_customers': df['customer_id'].nunique(),
            'unique_products': df['product_name'].nunique(),
            'date_range': f"{df['transaction_date'].min().strftime('%Y-%m-%d')} to {df['transaction_date'].max().strftime('%Y-%m-%d')}",
            'avg_customer_satisfaction': df['customer_satisfaction'].mean()
        }
        
        self.insights['basic_stats'] = basic_stats
        self.summary_stats = df.describe()
        
        print(f"   ✓ Total Transactions: {basic_stats['total_transactions']:,}")
        print(f"   ✓ Total Revenue: ${basic_stats['total_revenue']:,.2f}")
        print(f"   ✓ Average Order Value: ${basic_stats['avg_order_value']:.2f}")
        
    def _customer_analysis(self):
        """Analyze customer behavior and segmentation"""
        print("👥 Customer Behavior Analysis...")
        
        df = self.data
        
        # Customer segment performance
        segment_analysis = df.groupby('customer_segment').agg({
            'total_revenue': ['sum', 'mean', 'count'],
            'profit': 'sum',
            'customer_satisfaction': 'mean',
            'quantity': 'sum'
        }).round(2)
        
        # Top customers by revenue
        top_customers = df.groupby('customer_id').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).sort_values('total_revenue', ascending=False).head(10)
        
        # Customer lifetime value analysis
        ltv_analysis = df.groupby('customer_segment')['customer_ltv'].agg(['mean', 'median', 'std']).round(2)
        
        self.insights['customer_analysis'] = {
            'segment_performance': segment_analysis,
            'top_customers': top_customers,
            'ltv_analysis': ltv_analysis
        }
        
        print(f"   ✓ Analyzed {df['customer_id'].nunique():,} unique customers")
        print(f"   ✓ Customer segments: {', '.join(df['customer_segment'].unique())}")
    
    def _product_analysis(self):
        """Analyze product performance and categories"""
        print("📦 Product Performance Analysis...")
        
        df = self.data
        
        # Category performance
        category_performance = df.groupby('category').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'profit_margin_pct': 'mean',
            'customer_satisfaction': 'mean'
        }).sort_values('total_revenue', ascending=False).round(2)
        
        # Top products
        product_performance = df.groupby(['category', 'product_name']).agg({
            'total_revenue': 'sum',
            'quantity': 'sum',
            'profit': 'sum'
        }).sort_values('total_revenue', ascending=False).head(15)
        
        # Seasonal product analysis
        seasonal_products = df.groupby(['category', 'quarter'])['total_revenue'].sum().unstack(fill_value=0)
        
        self.insights['product_analysis'] = {
            'category_performance': category_performance,
            'top_products': product_performance,
            'seasonal_trends': seasonal_products
        }
        
        print(f"   ✓ Analyzed {df['category'].nunique()} product categories")
        print(f"   ✓ Top category by revenue: {category_performance.index[0]}")
    
    def _temporal_analysis(self):
        """Analyze trends over time"""
        print("📅 Temporal Trend Analysis...")
        
        df = self.data
        
        # Monthly trends
        monthly_trends = df.groupby(['year', 'month']).agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'transaction_id': 'count',
            'customer_satisfaction': 'mean'
        }).round(2)
        
        # Quarterly analysis
        quarterly_analysis = df.groupby('quarter').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).round(2)
        
        # Day of week patterns
        dow_patterns = df.groupby('day_of_week')['total_revenue'].agg(['sum', 'mean', 'count']).round(2)
        
        # Growth analysis
        monthly_revenue = df.groupby(df['transaction_date'].dt.to_period('M'))['total_revenue'].sum()
        growth_rate = ((monthly_revenue.iloc[-1] / monthly_revenue.iloc[0]) ** (1/len(monthly_revenue)) - 1) * 100
        
        self.insights['temporal_analysis'] = {
            'monthly_trends': monthly_trends,
            'quarterly_analysis': quarterly_analysis,
            'dow_patterns': dow_patterns,
            'growth_rate': growth_rate
        }
        
        print(f"   ✓ Analyzed trends across {df['year'].nunique()} years")
        print(f"   ✓ Monthly growth rate: {growth_rate:.2f}%")
    
    def _regional_analysis(self):
        """Analyze regional performance"""
        print("🗺️ Regional Performance Analysis...")
        
        df = self.data
        
        regional_performance = df.groupby('region').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'customer_satisfaction': 'mean',
            'customer_id': 'nunique'
        }).round(2)
        
        # Channel analysis
        channel_performance = df.groupby('channel').agg({
            'total_revenue': 'sum',
            'profit': 'sum',
            'customer_satisfaction': 'mean'
        }).round(2)
        
        self.insights['regional_analysis'] = {
            'regional_performance': regional_performance,
            'channel_performance': channel_performance
        }
        
        print(f"   ✓ Analyzed {df['region'].nunique()} regions")
        print(f"   ✓ Best performing region: {regional_performance.sort_values('total_revenue', ascending=False).index[0]}")
    
    def _advanced_insights(self):
        """Generate advanced business insights"""
        print("🎯 Advanced Business Insights...")
        
        df = self.data
        
        # Customer satisfaction correlation
        satisfaction_corr = df[['customer_satisfaction', 'total_revenue', 'profit_margin_pct', 'quantity']].corr()['customer_satisfaction'].round(3)
        
        # High-value transaction analysis
        high_value_analysis = df[df['high_value']].groupby('category')['total_revenue'].agg(['count', 'mean']).round(2)
        
        # Profitability insights
        profitability_by_segment = df.groupby(['customer_segment', 'category'])['profit_margin_pct'].mean().unstack(fill_value=0).round(2)
        
        self.insights['advanced_insights'] = {
            'satisfaction_correlation': satisfaction_corr,
            'high_value_transactions': high_value_analysis,
            'profitability_matrix': profitability_by_segment
        }
        
        print("   ✓ Generated correlation analysis")
        print("   ✓ Identified profitability patterns")
        
        print(f"\n✅ Analysis Complete! Generated {len(self.insights)} insight categories")

# ================================
# 3. VISUALIZATION DASHBOARD ENGINE
# ================================

class ECommerceVisualizer:
    """
    Professional visualization and dashboard creation
    Demonstrates advanced matplotlib/seaborn skills
    """
    
    def __init__(self, data, insights):
        """Initialize visualizer with data and analysis insights"""
        self.data = data
        self.insights = insights
        
        # Set professional styling
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
    def create_executive_dashboard(self):
        """Create comprehensive executive dashboard"""
        print("\n🎨 CREATING EXECUTIVE DASHBOARD")
        print("-" * 40)
        
        # Create main dashboard
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle('E-Commerce Business Intelligence Dashboard\nComprehensive Performance Analytics & Insights', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        # Define complex grid layout
        gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.35, wspace=0.3)
        
        self._plot_revenue_overview(fig, gs)
        self._plot_customer_segments(fig, gs) 
        self._plot_product_performance(fig, gs)
        self._plot_temporal_trends(fig, gs)
        self._plot_regional_analysis(fig, gs)
        self._plot_satisfaction_analysis(fig, gs)
        self._plot_profitability_matrix(fig, gs)
        self._plot_channel_performance(fig, gs)
        self._create_kpi_panel(fig, gs)
        
        plt.tight_layout()
        plt.savefig('ecommerce_executive_dashboard.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
        
        # Create additional detailed analysis
        self._create_detailed_analysis_plots()
        
        print("✅ Executive dashboard created and saved!")
    
    def _plot_revenue_overview(self, fig, gs):
        """Revenue and profit overview charts"""
        # Monthly revenue trend
        ax1 = fig.add_subplot(gs[0, :2])
        monthly_data = self.data.groupby(self.data['transaction_date'].dt.to_period('M')).agg({
            'total_revenue': 'sum',
            'profit': 'sum'
        })
        
        ax1.plot(monthly_data.index.astype(str), monthly_data['total_revenue'], 
                marker='o', linewidth=3, markersize=8, color='#E74C3C', label='Revenue')
        ax1.fill_between(monthly_data.index.astype(str), monthly_data['total_revenue'], 
                        alpha=0.3, color='#E74C3C')
        
        ax2 = ax1.twinx()
        ax2.plot(monthly_data.index.astype(str), monthly_data['profit'],
                marker='s', linewidth=3, markersize=6, color='#2ECC71', label='Profit')
        
        ax1.set_title('Monthly Revenue & Profit Trends', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Revenue ($)', color='#E74C3C', fontweight='bold')
        ax2.set_ylabel('Profit ($)', color='#2ECC71', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = range(len(monthly_data))
        z = np.polyfit(x_numeric, monthly_data['total_revenue'], 1)
        p = np.poly1d(z)
        ax1.plot(monthly_data.index.astype(str), p(x_numeric), 
                "--", color='darkred', alpha=0.8, linewidth=2)
    
    def _plot_customer_segments(self, fig, gs):
        """Customer segment analysis"""
        ax = fig.add_subplot(gs[0, 2:])
        segment_data = self.insights['customer_analysis']['segment_performance']
        
        # Extract revenue data
        revenue_data = segment_data['total_revenue']['sum']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(revenue_data.values, labels=revenue_data.index,
                                         autopct='%1.1f%%', startangle=90,
                                         colors=self.colors[:len(revenue_data)],
                                         explode=[0.05 if x == revenue_data.max() else 0 for x in revenue_data.values])
        
        ax.set_title('Revenue Distribution by Customer Segment', fontweight='bold', fontsize=14)
        
        # Add value annotations
        for i, (wedge, value) in enumerate(zip(wedges, revenue_data.values)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = wedge.r * 0.7 * np.cos(np.radians(angle))
            y = wedge.r * 0.7 * np.sin(np.radians(angle))
            ax.annotate(f'${value:,.0f}', (x, y), ha='center', va='center',
                       fontweight='bold', fontsize=10)
    
    def _plot_product_performance(self, fig, gs):
        """Product category performance"""
        ax = fig.add_subplot(gs[1, :2])
        category_data = self.insights['product_analysis']['category_performance'].sort_values('total_revenue', ascending=True)
        
        bars = ax.barh(category_data.index, category_data['total_revenue'], 
                      color=self.colors[:len(category_data)])
        ax.set_title('Product Category Performance', fontweight='bold', fontsize=14)
        ax.set_xlabel('Total Revenue ($)', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, category_data['total_revenue']):
            ax.text(bar.get_width() + max(category_data['total_revenue']) * 0.01,
                   bar.get_y() + bar.get_height()/2, f'${value:,.0f}',
                   va='center', fontweight='bold')
    
    def _plot_temporal_trends(self, fig, gs):
        """Quarterly and daily patterns"""
        # Quarterly performance
        ax1 = fig.add_subplot(gs[1, 2])
        quarterly_data = self.insights['temporal_analysis']['quarterly_analysis']
        
        bars = ax1.bar(quarterly_data.index, quarterly_data['total_revenue'],
                      color=['#3498DB', '#2ECC71', '#F39C12', '#E74C3C'])
        ax1.set_title('Quarterly Revenue', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Revenue ($)')
        
        for bar, value in zip(bars, quarterly_data['total_revenue']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quarterly_data['total_revenue'])*0.02,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Day of week patterns
        ax2 = fig.add_subplot(gs[1, 3])
        dow_data = self.insights['temporal_analysis']['dow_patterns']['mean'].reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        bars = ax2.bar(range(len(dow_data)), dow_data.values,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
        ax2.set_title('Daily Revenue Patterns', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Avg Revenue ($)')
        ax2.set_xticks(range(len(dow_data)))
        ax2.set_xticklabels([day[:3] for day in dow_data.index], rotation=45)
    
    def _plot_regional_analysis(self, fig, gs):
        """Regional and channel performance"""
        # Regional performance
        ax1 = fig.add_subplot(gs[2, :2])
        regional_data = self.insights['regional_analysis']['regional_performance']
        
        x = np.arange(len(regional_data))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, regional_data['total_revenue'], width, 
                       label='Revenue', color='#3498DB', alpha=0.8)
        bars2 = ax1.bar(x + width/2, regional_data['profit'], width,
                       label='Profit', color='#2ECC71', alpha=0.8)
        
        ax1.set_title('Regional Performance: Revenue vs Profit', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Amount ($)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regional_data.index)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    def _plot_satisfaction_analysis(self, fig, gs):
        """Customer satisfaction analysis"""
        ax = fig.add_subplot(gs[2, 2])
        
        # Satisfaction distribution
        ax.hist(self.data['customer_satisfaction'], bins=20, color='#9B59B6', alpha=0.7, edgecolor='black')
        ax.axvline(self.data['customer_satisfaction'].mean(), color='red', linestyle='--',
                  linewidth=3, label=f'Mean: {self.data["customer_satisfaction"].mean():.2f}')
        ax.set_title('Customer Satisfaction Distribution', fontweight='bold', fontsize=12)
        ax.set_xlabel('Satisfaction Rating')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_profitability_matrix(self, fig, gs):
        """Profitability heatmap"""
        ax = fig.add_subplot(gs[2, 3])
        
        profit_matrix = self.insights['advanced_insights']['profitability_matrix']
        sns.heatmap(profit_matrix, annot=True, cmap='RdYlGn', ax=ax, fmt='.1f',
                   cbar_kws={'label': 'Profit Margin %'})
        ax.set_title('Profitability by Segment & Category', fontweight='bold', fontsize=12)
        ax.set_xlabel('Category')
        ax.set_ylabel('Customer Segment')
    
    def _plot_channel_performance(self, fig, gs):
        """Sales channel analysis"""
        ax = fig.add_subplot(gs[3, :2])
        channel_data = self.insights['regional_analysis']['channel_performance'].sort_values('total_revenue', ascending=False)
        
        # Create bubble chart
        x = range(len(channel_data))
        y = channel_data['total_revenue']
        sizes = channel_data['profit'] / 100
        colors_list = self.colors[:len(channel_data)]
        
        scatter = ax.scatter(x, y, s=sizes, c=colors_list, alpha=0.7, edgecolors='black', linewidth=2)
        
        ax.set_title('Sales Channel Performance\n(Bubble size = Profit)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Total Revenue ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(channel_data.index, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (channel, revenue) in enumerate(zip(channel_data.index, channel_data['total_revenue'])):
            ax.annotate(f'${revenue:,.0f}', (i, revenue), ha='center', va='bottom',
                       fontweight='bold', fontsize=10)
    
    def _create_kpi_panel(self, fig, gs):
        """Create KPI dashboard panel"""
        ax = fig.add_subplot(gs[3, 2:])
        ax.axis('off')
        
        # Calculate KPIs
        basic_stats = self.insights['basic_stats']
        
        kpis = [
            ('Total Revenue', f"${basic_stats['total_revenue']:,.0f}", '#E74C3C'),
            ('Total Transactions', f"{basic_stats['total_transactions']:,}", '#3498DB'),
            ('Avg Order Value', f"${basic_stats['avg_order_value']:.2f}", '#2ECC71'),
            ('Unique Customers', f"{basic_stats['unique_customers']:,}", '#F39C12'),
            ('Avg Satisfaction', f"{basic_stats['avg_customer_satisfaction']:.2f}/5.0", '#9B59B6'),
            ('Profit Margin', f"{basic_stats['avg_profit_margin']:.1f}%", '#1ABC9C')
        ]
        
        # Create KPI boxes
        kpi_width = 1.0 / len(kpis)
        for i, (title, value, color) in enumerate(kpis):
            x_pos = i * kpi_width + kpi_width/2
            
            # Create colored box
            box = plt.Rectangle((i * kpi_width + 0.01, 0.3), kpi_width - 0.02, 0.4,
                               facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(box)
            
            # Add text
            ax.text(x_pos, 0.6, title, ha='center', va='center', fontweight='bold', fontsize=12)
            ax.text(x_pos, 0.4, value, ha='center', va='center', fontweight='bold', fontsize=14, color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Key Performance Indicators', fontweight='bold', fontsize=16, y=0.9)
    
    def _create_detailed_analysis_plots(self):
        """Create additional detailed analysis visualizations"""
        print("📈 Creating detailed analysis plots...")
        
        # Create detailed analysis figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Detailed E-Commerce Analysis\nAdvanced Statistical Insights', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Revenue vs Satisfaction Scatter
        axes[0, 0].scatter(self.data['customer_satisfaction'], self.data['total_revenue'],
                          alpha=0.6, c=self.data['profit'], cmap='viridis', s=30)
        axes[0, 0].set_xlabel('Customer Satisfaction')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].set_title('Revenue vs Customer Satisfaction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Product quantity distribution
        for i, category in enumerate(self.data['category'].unique()):
            cat_data = self.data[self.data['category'] == category]['quantity']
            axes[0, 1].hist(cat_data, alpha=0.6, label=category, bins=15)
        axes[0, 1].set_xlabel('Quantity per Transaction')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Quantity Distribution by Category')
        axes[0, 1].legend()
        
        # 3. Customer segment satisfaction
        sns.boxplot(data=self.data, x='customer_segment', y='customer_satisfaction', ax=axes[0, 2])
        axes[0, 2].set_title('Satisfaction by Customer Segment')
        axes[0, 2].set_xlabel('Customer Segment')
        
        # 4. Monthly growth trend
        monthly_revenue = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['total_revenue'].sum()
        growth_rates = monthly_revenue.pct_change() * 100
        axes[1, 0].plot(growth_rates.index.astype(str), growth_rates.values, marker='o', linewidth=2)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Monthly Growth Rate')
        axes[1, 0].set_ylabel('Growth Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Channel vs Category heatmap
        channel_category = self.data.groupby(['channel', 'category'])['total_revenue'].sum().unstack(fill_value=0)
        sns.heatmap(channel_category, annot=True, fmt='.0f', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Revenue: Channel vs Category')
        
        # 6. High-value transactions
        high_value_by_category = self.data[self.data['high_value']].groupby('category').size()
        axes[1, 2].pie(high_value_by_category.values, labels=high_value_by_category.index,
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('High-Value Transactions by Category')
        
        # 7. Customer lifetime value distribution
        ltv_by_segment = [self.data[self.data['customer_segment'] == seg]['customer_ltv'] 
                         for seg in self.data['customer_segment'].unique()]
        axes[2, 0].boxplot(ltv_by_segment, labels=self.data['customer_segment'].unique())
        axes[2, 0].set_title('Customer Lifetime Value by Segment')
        axes[2, 0].set_ylabel('Customer LTV ($)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Profit margin trends
        monthly_margin = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['profit_margin_pct'].mean()
        axes[2, 1].plot(monthly_margin.index.astype(str), monthly_margin.values, 
                       marker='s', linewidth=2, color='green')
        axes[2, 1].set_title('Monthly Profit Margin Trends')
        axes[2, 1].set_ylabel('Profit Margin (%)')
        axes[2, 1].tick_params(axis='x', rotation=45)
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Regional customer satisfaction
        regional_satisfaction = self.data.groupby('region')['customer_satisfaction'].mean().sort_values(ascending=True)
        bars = axes[2, 2].barh(regional_satisfaction.index, regional_satisfaction.values,
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[2, 2].set_title('Average Satisfaction by Region')
        axes[2, 2].set_xlabel('Average Satisfaction Rating')
        
        # Add value labels
        for bar, value in zip(bars, regional_satisfaction.values):
            axes[2, 2].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                           f'{value:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ecommerce_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Detailed analysis plots created!")

# ================================
# 4. BUSINESS INTELLIGENCE REPORTER
# ================================

class BusinessIntelligenceReporter:
    """
    Generate comprehensive business intelligence reports
    Demonstrates data interpretation and business communication skills
    """
    
    def __init__(self, data, insights):
        self.data = data
        self.insights = insights
        self.report = {}
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        print("\n📋 GENERATING BUSINESS INTELLIGENCE REPORT")
        print("-" * 50)
        
        basic_stats = self.insights['basic_stats']
        
        # Executive Summary
        executive_summary = {
            'overview': f"""
            E-Commerce Performance Summary ({basic_stats['date_range']})
            ================================================================
            
            BUSINESS PERFORMANCE HIGHLIGHTS:
            • Total Revenue: ${basic_stats['total_revenue']:,.2f}
            • Total Transactions: {basic_stats['total_transactions']:,}
            • Average Order Value: ${basic_stats['avg_order_value']:.2f}
            • Customer Satisfaction: {basic_stats['avg_customer_satisfaction']:.2f}/5.0
            • Unique Customers Served: {basic_stats['unique_customers']:,}
            • Average Profit Margin: {basic_stats['avg_profit_margin']:.1f}%
            """,
            
            'key_insights': self._generate_key_insights(),
            'recommendations': self._generate_recommendations(),
            'risk_assessment': self._generate_risk_assessment()
        }
        
        self.report = executive_summary
        return executive_summary
    
    def _generate_key_insights(self):
        """Generate key business insights"""
        insights = []
        
        # Customer segment analysis
        segment_perf = self.insights['customer_analysis']['segment_performance']
        top_segment = segment_perf['total_revenue']['sum'].idxmax()
        top_segment_revenue = segment_perf['total_revenue']['sum'][top_segment]
        total_revenue = segment_perf['total_revenue']['sum'].sum()
        
        insights.append(f"🎯 CUSTOMER INSIGHTS: {top_segment} customers drive {(top_segment_revenue/total_revenue)*100:.1f}% of total revenue")
        
        # Product performance
        category_perf = self.insights['product_analysis']['category_performance']
        top_category = category_perf['total_revenue'].idxmax()
        top_category_margin = category_perf['profit_margin_pct'][top_category]
        
        insights.append(f"📦 PRODUCT INSIGHTS: {top_category} is the top revenue category with {top_category_margin:.1f}% profit margin")
        
        # Temporal insights
        quarterly_data = self.insights['temporal_analysis']['quarterly_analysis']
        best_quarter = quarterly_data['total_revenue'].idxmax()
        growth_rate = self.insights['temporal_analysis']['growth_rate']
        
        insights.append(f"📈 GROWTH INSIGHTS: {best_quarter} shows strongest performance with {growth_rate:.1f}% monthly growth rate")
        
        # Regional insights  
        regional_perf = self.insights['regional_analysis']['regional_performance']
        top_region = regional_perf['total_revenue'].idxmax()
        
        insights.append(f"🗺️ REGIONAL INSIGHTS: {top_region} region leads with highest revenue and customer base")
        
        # Satisfaction insights
        satisfaction_corr = self.insights['advanced_insights']['satisfaction_correlation']
        strongest_corr = satisfaction_corr.drop('customer_satisfaction').abs().idxmax()
        
        insights.append(f"⭐ SATISFACTION INSIGHTS: Customer satisfaction most strongly correlates with {strongest_corr}")
        
        return insights
    
    def _generate_recommendations(self):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Based on customer segments
        segment_perf = self.insights['customer_analysis']['segment_performance']
        premium_revenue = segment_perf['total_revenue']['sum']['Premium']
        total_revenue = segment_perf['total_revenue']['sum'].sum()
        
        if (premium_revenue / total_revenue) > 0.3:
            recommendations.append("💎 Focus on premium customer retention programs and exclusive offerings")
        
        # Based on product performance
        category_perf = self.insights['product_analysis']['category_performance']
        lowest_margin_category = category_perf['profit_margin_pct'].idxmin()
        
        recommendations.append(f"📊 Optimize pricing strategy for {lowest_margin_category} category to improve profitability")
        
        # Based on channels
        channel_perf = self.insights['regional_analysis']['channel_performance']
        mobile_revenue = channel_perf.loc['Mobile App', 'total_revenue'] if 'Mobile App' in channel_perf.index else 0
        total_channel_revenue = channel_perf['total_revenue'].sum()
        
        if (mobile_revenue / total_channel_revenue) < 0.3:
            recommendations.append("📱 Invest in mobile app optimization to capture growing mobile commerce trend")
        
        # Based on satisfaction
        avg_satisfaction = self.insights['basic_stats']['avg_customer_satisfaction']
        if avg_satisfaction < 4.0:
            recommendations.append("🎯 Implement customer experience improvement initiatives to boost satisfaction scores")
        
        recommendations.append("📈 Develop seasonal marketing campaigns based on quarterly performance patterns")
        
        return recommendations
    
    def _generate_risk_assessment(self):
        """Generate risk assessment and mitigation strategies"""
        risks = []
        
        # Customer concentration risk
        segment_perf = self.insights['customer_analysis']['segment_performance']
        premium_ratio = segment_perf['total_revenue']['sum']['Premium'] / segment_perf['total_revenue']['sum'].sum()
        
        if premium_ratio > 0.4:
            risks.append("⚠️ HIGH CUSTOMER CONCENTRATION: Over-dependence on premium segment customers")
        
        # Seasonal risk
        quarterly_data = self.insights['temporal_analysis']['quarterly_analysis']
        q4_ratio = quarterly_data.loc['Q4', 'total_revenue'] / quarterly_data['total_revenue'].sum()
        
        if q4_ratio > 0.35:
            risks.append("📅 SEASONAL DEPENDENCY: High reliance on Q4 performance may create cash flow risks")
        
        # Profitability risk
        avg_margin = self.insights['basic_stats']['avg_profit_margin']
        if avg_margin < 25:
            risks.append("💰 MARGIN PRESSURE: Below-average profit margins indicate pricing or cost challenges")
        
        # Geographic risk
        regional_perf = self.insights['regional_analysis']['regional_performance']
        top_region_ratio = regional_perf['total_revenue'].max() / regional_perf['total_revenue'].sum()
        
        if top_region_ratio > 0.35:
            risks.append("🗺️ GEOGRAPHIC CONCENTRATION: Over-reliance on single region creates market risk")
        
        return risks
    
    def export_reports(self):
        """Export comprehensive reports to files"""
        print("\n💾 Exporting Business Intelligence Reports...")

        # Export executive summary
        with open('executive_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(self.report['overview'])
            f.write("\n\nKEY BUSINESS INSIGHTS:\n")
            for insight in self.report['key_insights']:
                f.write(f"• {insight}\n")
            
            f.write("\n\nSTRATEGIC RECOMMENDATIONS:\n")
            for rec in self.report['recommendations']:
                f.write(f"• {rec}\n")
            
            f.write("\n\nRISK ASSESSMENT:\n")
            for risk in self.report['risk_assessment']:
                f.write(f"• {risk}\n")

        # Export detailed insights as JSON
        detailed_insights = {
            'basic_statistics': self.insights['basic_stats'],
            'customer_analysis': self._serialize_dataframes(self.insights['customer_analysis']),
            'product_analysis': self._serialize_dataframes(self.insights['product_analysis']),
            'temporal_analysis': self._serialize_dataframes(self.insights['temporal_analysis']),
            'regional_analysis': self._serialize_dataframes(self.insights['regional_analysis'])
        }

        with open('detailed_insights.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_insights, f, indent=2, default=str)

        print("✅ Reports exported successfully!")
        print("   • executive_summary_report.txt")
        print("   • detailed_insights.json")
    
    def _serialize_dataframes(self, dataframes_dict):
        result = {}
        for key, df in dataframes_dict.items():
            if isinstance(df, pd.DataFrame):
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.copy()
                    df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
                df = df.reset_index()
                result[key] = df.to_dict(orient="records")
            else:
                # For scalars or other types, just store the value
                result[key] = df
        return result

# ================================
# 5. MAIN EXECUTION ENGINE
# ================================

def main():
    """
    Main execution function - demonstrates complete data science pipeline
    """
    print("🎯 EXECUTING COMPLETE DATA SCIENCE PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Data Generation
        print("\n🔥 STEP 1: DATA GENERATION")
        data_generator = ECommerceDataGenerator(seed=42)
        raw_data = data_generator.generate_transactions(n_records=10000, date_range_days=730)
        
        print(f"\n📊 Generated Dataset Overview:")
        print(f"   • Shape: {raw_data.shape}")
        print(f"   • Columns: {list(raw_data.columns)}")
        print(f"   • Date Range: {raw_data['transaction_date'].min()} to {raw_data['transaction_date'].max()}")
        
        # Step 2: Data Analysis
        print("\n🔥 STEP 2: COMPREHENSIVE ANALYSIS")
        analyzer = ECommerceAnalyzer(raw_data)
        insights = analyzer.perform_comprehensive_analysis()
        
        # Step 3: Visualization
        print("\n🔥 STEP 3: VISUALIZATION & DASHBOARD")
        visualizer = ECommerceVisualizer(raw_data, insights)
        visualizer.create_executive_dashboard()
        
        # Step 4: Business Intelligence
        print("\n🔥 STEP 4: BUSINESS INTELLIGENCE REPORTING")
        reporter = BusinessIntelligenceReporter(raw_data, insights)
        executive_summary = reporter.generate_executive_summary()
        reporter.export_reports()
        
        # Step 5: Data Export
        print("\n🔥 STEP 5: DATA EXPORT & DOCUMENTATION")
        raw_data.to_csv('ecommerce_complete_dataset.csv', index=False)
        raw_data.describe().to_csv('dataset_summary_statistics.csv')
        
        # Final Summary
        print("\n" + "="*60)
        print("🎉 PORTFOLIO PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n🏆 PROJECT ACHIEVEMENTS:")
        achievements = [
            "✅ Generated 10,000 realistic e-commerce transactions",
            "✅ Built comprehensive OOP-based analysis framework", 
            "✅ Performed advanced statistical analysis with Pandas/NumPy",
            "✅ Created professional executive dashboard (20+ visualizations)",
            "✅ Generated actionable business intelligence insights",
            "✅ Demonstrated end-to-end data science pipeline",
            "✅ Applied all Week 1 concepts in integrated project",
            "✅ Created publication-ready visualizations and reports"
        ]
        
        for achievement in achievements:
            print(f"   {achievement}")
        
        print("\n📁 FILES GENERATED:")
        files = [
            "• ecommerce_executive_dashboard.png - Main BI dashboard",
            "• ecommerce_detailed_analysis.png - Detailed analysis plots", 
            "• ecommerce_complete_dataset.csv - Full generated dataset",
            "• dataset_summary_statistics.csv - Statistical summary",
            "• executive_summary_report.txt - Business intelligence report",
            "• detailed_insights.json - Comprehensive insights data"
        ]
        
        for file_info in files:
            print(f"   {file_info}")
        
        print("\n💪 SKILLS DEMONSTRATED:")
        skills = [
            "Python Programming: OOP, functions, error handling, data structures",
            "Data Science: NumPy arrays, Pandas DataFrames, statistical analysis", 
            "Data Visualization: Matplotlib, Seaborn, dashboard design",
            "Business Intelligence: KPIs, insights generation, reporting",
            "Software Engineering: Code organization, documentation, modularity",
            "Problem Solving: End-to-end pipeline, integration, optimization"
        ]
        
        for skill in skills:
            print(f"   • {skill}")
        
        print(f"\n🎯 PORTFOLIO READINESS: PROFESSIONAL LEVEL")
        print("   Ready for Data Analyst positions (10-15 LPA)")
        print("   Demonstrates business-ready technical capabilities")
        print("   Shows progression from basics to advanced integration")
        
        print(f"\n🚀 WEEK 1 STATUS: COMPLETE!")
        print("   Foundation solidly established for Week 2 ML concepts")
        print("   Portfolio quality exceeds entry-level expectations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {str(e)}")
        print("🔧 Debugging information:")
        import traceback
        traceback.print_exc()
        return False

# ================================
# EXECUTE THE PROJECT
# ================================

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "🎊" * 20)
        print("  CONGRATULATIONS!")
        print("  Week 1 Portfolio Project Complete!")
        print("  Ready for Week 2: Machine Learning!")
        print("🎊" * 20)
    else:
        print("\n⚠️ Project execution encountered issues.")
        print("💡 Review error messages and debug systematically.")