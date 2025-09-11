# Day 4 Mini-Project: Smart Data Analysis System
# Combining Data Structures, List Comprehensions, and Algorithms

import json
import csv
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

class DataAnalysisSystem:
    """Smart system for analyzing various types of data using advanced data structures"""
    
    def __init__(self, system_name="AI/ML Learning Analytics"):
        self.system_name = system_name
        self.datasets = {}  # Store multiple datasets
        self.analysis_cache = {}  # Cache results for performance
        self.metadata = {
            "created": datetime.now().isoformat(),
            "total_analyses": 0,
            "datasets_processed": 0
        }
    
    def load_dataset(self, name, data, data_type="generic"):
        """Load dataset with automatic type detection and validation"""
        if not isinstance(data, (list, dict)):
            raise ValueError("Data must be a list or dictionary")
        
        # Validate and process data
        processed_data = self._validate_and_process(data, data_type)
        
        self.datasets[name] = {
            "data": processed_data,
            "type": data_type,
            "loaded_at": datetime.now().isoformat(),
            "size": len(processed_data),
            "columns": self._extract_columns(processed_data)
        }
        
        self.metadata["datasets_processed"] += 1
        print(f"Dataset '{name}' loaded successfully. Size: {len(processed_data)} records")
        
        return True
    
    def _validate_and_process(self, data, data_type):
        """Validate and clean input data"""
        if data_type == "student_performance":
            # Validate student performance data
            required_fields = ["name", "scores", "courses"]
            for record in data:
                if not all(field in record for field in required_fields):
                    raise ValueError(f"Student record missing required fields: {required_fields}")
        
        elif data_type == "learning_progress":
            # Validate learning progress data
            for record in data:
                if "day" not in record or "topics" not in record:
                    raise ValueError("Learning progress must have 'day' and 'topics' fields")
        
        # Clean data using list comprehensions
        cleaned_data = [
            {k: v for k, v in record.items() if v is not None}
            for record in data
            if record  # Remove empty records
        ]
        
        return cleaned_data
    
    def _extract_columns(self, data):
        """Extract column names from dataset"""
        if not data:
            return []
        
        if isinstance(data[0], dict):
            # Get all unique keys across all records
            all_keys = set()
            for record in data:
                all_keys.update(record.keys())
            return sorted(list(all_keys))
        
        return ["value"]  # For simple lists
    
    def analyze_dataset(self, dataset_name, analysis_type="comprehensive"):
        """Perform various types of analysis on dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Check cache first
        cache_key = f"{dataset_name}_{analysis_type}"
        if cache_key in self.analysis_cache:
            print(f"Returning cached analysis for {dataset_name}")
            return self.analysis_cache[cache_key]
        
        dataset = self.datasets[dataset_name]
        data = dataset["data"]
        
        if analysis_type == "comprehensive":
            results = self._comprehensive_analysis(data, dataset)
        elif analysis_type == "performance":
            results = self._performance_analysis(data)
        elif analysis_type == "learning_patterns":
            results = self._learning_pattern_analysis(data)
        else:
            results = self._basic_statistics(data)
        
        # Cache results
        self.analysis_cache[cache_key] = results
        self.metadata["total_analyses"] += 1
        
        return results
    
    def _comprehensive_analysis(self, data, dataset_info):
        """Comprehensive analysis using various data structures and algorithms"""
        analysis = {
            "dataset_info": {
                "name": dataset_info.get("type", "unknown"),
                "size": len(data),
                "columns": dataset_info["columns"]
            },
            "basic_stats": {},
            "patterns": {},
            "insights": []
        }
        
        if not data:
            return analysis
        
        # Numeric analysis using list comprehensions and built-in functions
        numeric_fields = self._find_numeric_fields(data)
        
        for field in numeric_fields:
            values = [record[field] for record in data if field in record and isinstance(record[field], (int, float))]
            
            if values:
                analysis["basic_stats"][field] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values)
                }
        
        # Categorical analysis using Counter and defaultdict
        categorical_fields = self._find_categorical_fields(data)
        
        for field in categorical_fields:
            values = [record[field] for record in data if field in record]
            counter = Counter(values)
            
            analysis["patterns"][field] = {
                "unique_values": len(counter),
                "most_common": counter.most_common(5),
                "distribution": dict(counter)
            }
        
        # Generate insights using advanced algorithms
        analysis["insights"] = self._generate_insights(data, analysis)
        
        return analysis
    
    def _performance_analysis(self, data):
        """Analyze student/learning performance data"""
        performance_metrics = {
            "overall_performance": {},
            "subject_analysis": {},
            "improvement_trends": {},
            "recommendations": []
        }
        
        # Extract performance scores using list comprehensions
        all_scores = []
        subject_scores = defaultdict(list)
        
        for record in data:
            if "scores" in record:
                if isinstance(record["scores"], dict):
                    # Subject-wise scores
                    for subject, score in record["scores"].items():
                        subject_scores[subject].append(score)
                        all_scores.append(score)
                elif isinstance(record["scores"], (int, float)):
                    all_scores.append(record["scores"])
        
        # Overall performance statistics
        if all_scores:
            performance_metrics["overall_performance"] = {
                "average_score": statistics.mean(all_scores),
                "median_score": statistics.median(all_scores),
                "score_distribution": {
                    "excellent (90+)": len([s for s in all_scores if s >= 90]),
                    "good (80-89)": len([s for s in all_scores if 80 <= s < 90]),
                    "average (70-79)": len([s for s in all_scores if 70 <= s < 80]),
                    "below_average (<70)": len([s for s in all_scores if s < 70])
                },
                "total_assessments": len(all_scores)
            }
        
        # Subject-wise analysis
        for subject, scores in subject_scores.items():
            if scores:
                performance_metrics["subject_analysis"][subject] = {
                    "average": statistics.mean(scores),
                    "difficulty_level": "Hard" if statistics.mean(scores) < 75 else "Medium" if statistics.mean(scores) < 85 else "Easy",
                    "consistency": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "top_score": max(scores),
                    "improvement_needed": statistics.mean(scores) < 80
                }
        
        # Generate recommendations using algorithmic logic
        performance_metrics["recommendations"] = self._generate_performance_recommendations(
            performance_metrics["overall_performance"],
            performance_metrics["subject_analysis"]
        )
        
        return performance_metrics
    
    def _learning_pattern_analysis(self, data):
        """Analyze learning patterns and progress over time"""
        patterns = {
            "daily_progress": {},
            "skill_development": {},
            "learning_velocity": {},
            "predictions": {}
        }
        
        # Sort data by day if available
        if data and "day" in data[0]:
            sorted_data = sorted(data, key=lambda x: x.get("day", 0))
        else:
            sorted_data = data
        
        # Analyze daily progress
        daily_topics = defaultdict(list)
        daily_hours = defaultdict(float)
        
        for record in sorted_data:
            day = record.get("day", 0)
            topics = record.get("topics", [])
            hours = record.get("hours", 0)
            
            daily_topics[day].extend(topics if isinstance(topics, list) else [topics])
            daily_hours[day] += hours
        
        # Calculate learning velocity (topics per day)
        total_days = len(daily_topics)
        total_topics = sum(len(topics) for topics in daily_topics.values())
        
        patterns["learning_velocity"] = {
            "topics_per_day": total_topics / total_days if total_days > 0 else 0,
            "hours_per_day": sum(daily_hours.values()) / total_days if total_days > 0 else 0,
            "most_productive_day": max(daily_hours, key=daily_hours.get) if daily_hours else None,
            "consistency_score": self._calculate_consistency(list(daily_hours.values()))
        }
        
        # Skill development tracking
        all_topics = [topic for topics in daily_topics.values() for topic in topics]
        topic_frequency = Counter(all_topics)
        
        patterns["skill_development"] = {
            "total_unique_topics": len(topic_frequency),
            "most_practiced": topic_frequency.most_common(10),
            "learning_breadth": len(topic_frequency) / total_topics if total_topics > 0 else 0
        }
        
        # Simple prediction based on current trends
        if len(daily_hours) >= 3:
            recent_performance = list(daily_hours.values())[-3:]
            trend = "Improving" if recent_performance[-1] > recent_performance[0] else "Declining"
            
            patterns["predictions"] = {
                "trend": trend,
                "projected_30_day_topics": int((total_topics / total_days) * 30) if total_days > 0 else 0,
                "estimated_completion_time": max(90 - total_days, 0) if total_days > 0 else 90
            }
        
        return patterns
    
    def _find_numeric_fields(self, data):
        """Find numeric fields in dataset using type checking"""
        if not data:
            return []
        
        numeric_fields = set()
        for record in data:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_fields.add(key)
        
        return list(numeric_fields)
    
    def _find_categorical_fields(self, data):
        """Find categorical (string) fields in dataset"""
        if not data:
            return []
        
        categorical_fields = set()
        for record in data:
            for key, value in record.items():
                if isinstance(value, str):
                    categorical_fields.add(key)
        
        return list(categorical_fields)
    
    def _generate_insights(self, data, analysis):
        """Generate actionable insights using algorithmic analysis"""
        insights = []
        
        # Analyze basic statistics for insights
        for field, stats in analysis["basic_stats"].items():
            if stats["std_dev"] > stats["mean"] * 0.5:
                insights.append(f"High variability detected in {field} - consider investigating outliers")
            
            if stats["mean"] < 50:  # Assuming scores out of 100
                insights.append(f"Low average performance in {field} - additional focus needed")
        
        # Analyze patterns for insights
        for field, pattern in analysis["patterns"].items():
            if pattern["unique_values"] == 1:
                insights.append(f"No variation in {field} - all records have the same value")
            
            # Check for data quality issues
            most_common_count = pattern["most_common"][0][1] if pattern["most_common"] else 0
            if most_common_count > len(data) * 0.8:
                insights.append(f"Potential data quality issue: 80%+ of records have same {field} value")
        
        # Performance-based insights
        if len(analysis["basic_stats"]) > 0:
            best_field = max(analysis["basic_stats"], key=lambda x: analysis["basic_stats"][x]["mean"])
            worst_field = min(analysis["basic_stats"], key=lambda x: analysis["basic_stats"][x]["mean"])
            
            insights.append(f"Strongest performance area: {best_field}")
            insights.append(f"Area needing improvement: {worst_field}")
        
        return insights[:10]  # Return top 10 insights
    
    def _generate_performance_recommendations(self, overall, subjects):
        """Generate performance recommendations using rule-based algorithm"""
        recommendations = []
        
        # Overall performance recommendations
        if overall and overall["average_score"] < 75:
            recommendations.append("Overall performance below target - implement comprehensive study plan")
        
        if overall and overall["score_distribution"]["below_average (<70)"] > overall["total_assessments"] * 0.3:
            recommendations.append("High number of low scores - focus on fundamental concepts")
        
        # Subject-specific recommendations
        for subject, metrics in subjects.items():
            if metrics["improvement_needed"]:
                recommendations.append(f"Priority focus needed in {subject} (current avg: {metrics['average']:.1f})")
            
            if metrics["consistency"] > 15:  # High standard deviation
                recommendations.append(f"Inconsistent performance in {subject} - establish regular practice routine")
        
        # Learning strategy recommendations
        if subjects:
            easiest_subject = min(subjects, key=lambda x: subjects[x]["average"] if not subjects[x]["improvement_needed"] else 100)
            hardest_subject = max(subjects, key=lambda x: subjects[x]["average"] if subjects[x]["improvement_needed"] else 0)
            
            recommendations.append(f"Leverage strength in {easiest_subject} to build confidence")
            recommendations.append(f"Allocate extra study time to {hardest_subject}")
        
        return recommendations
    
    def _calculate_consistency(self, values):
        """Calculate consistency score (lower standard deviation = higher consistency)"""
        if len(values) < 2:
            return 100
        
        std_dev = statistics.stdev(values)
        mean_val = statistics.mean(values)
        
        # Normalize to 0-100 scale (higher is more consistent)
        coefficient_of_variation = std_dev / mean_val if mean_val > 0 else 0
        consistency = max(0, 100 - (coefficient_of_variation * 100))
        
        return round(consistency, 2)
    
    def compare_datasets(self, dataset1_name, dataset2_name):
        """Compare two datasets using statistical methods"""
        if dataset1_name not in self.datasets or dataset2_name not in self.datasets:
            raise ValueError("One or both datasets not found")
        
        analysis1 = self.analyze_dataset(dataset1_name)
        analysis2 = self.analyze_dataset(dataset2_name)
        
        comparison = {
            "dataset_comparison": {
                "dataset1": {
                    "name": dataset1_name,
                    "size": analysis1["dataset_info"]["size"]
                },
                "dataset2": {
                    "name": dataset2_name, 
                    "size": analysis2["dataset_info"]["size"]
                }
            },
            "statistical_differences": {},
            "insights": []
        }
        
        # Compare common numeric fields
        common_fields = set(analysis1["basic_stats"].keys()) & set(analysis2["basic_stats"].keys())
        
        for field in common_fields:
            stats1 = analysis1["basic_stats"][field]
            stats2 = analysis2["basic_stats"][field]
            
            comparison["statistical_differences"][field] = {
                "mean_difference": stats2["mean"] - stats1["mean"],
                "median_difference": stats2["median"] - stats1["median"],
                "variance_ratio": stats2["std_dev"] / stats1["std_dev"] if stats1["std_dev"] > 0 else "undefined"
            }
        
        # Generate comparison insights
        for field, diff in comparison["statistical_differences"].items():
            if abs(diff["mean_difference"]) > 5:  # Significant difference threshold
                direction = "higher" if diff["mean_difference"] > 0 else "lower"
                comparison["insights"].append(
                    f"{dataset2_name} has {direction} average {field} than {dataset1_name} "
                    f"(difference: {diff['mean_difference']:.2f})"
                )
        
        return comparison
    
    def export_analysis(self, filename, analyses=None):
        """Export analysis results to JSON file"""
        if analyses is None:
            analyses = {name: self.analyze_dataset(name) for name in self.datasets.keys()}
        
        export_data = {
            "system_info": {
                "name": self.system_name,
                "export_time": datetime.now().isoformat(),
                "metadata": self.metadata
            },
            "datasets": {name: info for name, info in self.datasets.items()},
            "analyses": analyses
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Analysis exported to {filename}")
    
    def get_system_summary(self):
        """Get comprehensive system summary"""
        summary = {
            "system_name": self.system_name,
            "datasets_loaded": len(self.datasets),
            "total_records": sum(dataset["size"] for dataset in self.datasets.values()),
            "analyses_performed": self.metadata["total_analyses"],
            "cache_size": len(self.analysis_cache),
            "dataset_details": []
        }
        
        for name, dataset in self.datasets.items():
            summary["dataset_details"].append({
                "name": name,
                "type": dataset["type"],
                "size": dataset["size"],
                "columns": len(dataset["columns"]),
                "loaded_at": dataset["loaded_at"]
            })
        
        return summary

# DEMONSTRATION AND TESTING
def demonstrate_data_analysis_system():
    """Demonstrate the complete data analysis system"""
    print("=== Smart Data Analysis System Demo ===")
    
    # Initialize system
    analyzer = DataAnalysisSystem("AI/ML Learning Analytics Pro")
    
    # Create sample student performance data
    student_data = [
        {"name": "Ashwin", "scores": {"python": 85, "ml": 78, "stats": 92}, "courses": ["Python", "ML", "Statistics"], "day": 3},
        {"name": "Priya", "scores": {"python": 92, "ml": 88, "stats": 85}, "courses": ["Python", "ML", "Statistics"], "day": 3},
        {"name": "Rahul", "scores": {"python": 76, "ml": 82, "stats": 79}, "courses": ["Python", "ML"], "day": 2},
        {"name": "Anita", "scores": {"python": 94, "ml": 91, "stats": 88}, "courses": ["Python", "ML", "Statistics"], "day": 3},
        {"name": "Vikram", "scores": {"python": 68, "ml": 72, "stats": 75}, "courses": ["Python", "Statistics"], "day": 2}
    ]
    
    # Create learning progress data
    learning_data = [
        {"day": 1, "topics": ["variables", "operators"], "hours": 3, "difficulty": "easy"},
        {"day": 2, "topics": ["loops", "functions"], "hours": 3.5, "difficulty": "medium"},
        {"day": 3, "topics": ["OOP", "file_handling"], "hours": 4, "difficulty": "hard"},
        {"day": 4, "topics": ["data_structures", "algorithms"], "hours": 3.5, "difficulty": "medium"}
    ]
    
    try:
        # Load datasets
        analyzer.load_dataset("student_performance", student_data, "student_performance")
        analyzer.load_dataset("learning_progress", learning_data, "learning_progress")
        
        print("\n1. COMPREHENSIVE ANALYSIS")
        print("-" * 40)
        
        # Analyze student performance
        performance_analysis = analyzer.analyze_dataset("student_performance", "performance")
        
        print("Overall Performance:")
        overall = performance_analysis["overall_performance"]
        print(f"  Average Score: {overall['average_score']:.1f}")
        print(f"  Total Assessments: {overall['total_assessments']}")
        
        print("\nSubject Analysis:")
        for subject, metrics in performance_analysis["subject_analysis"].items():
            print(f"  {subject}: Avg {metrics['average']:.1f} ({metrics['difficulty_level']})")
        
        print("\nRecommendations:")
        for rec in performance_analysis["recommendations"][:3]:
            print(f"  • {rec}")
        
        print("\n2. LEARNING PATTERN ANALYSIS")
        print("-" * 40)
        
        # Analyze learning patterns
        pattern_analysis = analyzer.analyze_dataset("learning_progress", "learning_patterns")
        
        velocity = pattern_analysis["learning_velocity"]
        print(f"Learning Velocity: {velocity['topics_per_day']:.1f} topics/day")
        print(f"Study Time: {velocity['hours_per_day']:.1f} hours/day")
        print(f"Consistency Score: {velocity['consistency_score']:.1f}%")
        
        development = pattern_analysis["skill_development"]
        print(f"Total Topics Covered: {development['total_unique_topics']}")
        print("Most Practiced Topics:")
        for topic, count in development["most_practiced"][:3]:
            print(f"  • {topic}: {count} times")
        
        if "predictions" in pattern_analysis:
            predictions = pattern_analysis["predictions"]
            print(f"\nTrend: {predictions['trend']}")
            print(f"30-day Projection: {predictions['projected_30_day_topics']} topics")
        
        print("\n3. SYSTEM SUMMARY")
        print("-" * 40)
        
        summary = analyzer.get_system_summary()
        print(f"System: {summary['system_name']}")
        print(f"Datasets: {summary['datasets_loaded']}")
        print(f"Total Records: {summary['total_records']}")
        print(f"Analyses Performed: {summary['analyses_performed']}")
        
        # Export results
        analyzer.export_analysis("analysis_results.json")
        
        print("\n=== Demo Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    demonstrate_data_analysis_system()
