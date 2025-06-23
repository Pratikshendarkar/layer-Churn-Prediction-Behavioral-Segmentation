# player_churn_analysis.py
# Complete Player Churn Prediction Analysis for Rockstar Analytics Portfolio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

# Optional imports with fallbacks
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠️  SMOTE not available. Using alternative resampling method.")
    SMOTE_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available. Using sklearn models only.")
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Using sklearn feature importance.")
    SHAP_AVAILABLE = False

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PlayerChurnAnalysis:
    """
    Complete Player Churn Prediction Analysis
    
    This class handles the entire machine learning pipeline for predicting
    player churn in gaming environments, specifically designed for
    Rockstar Analytics team applications.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_importance = {}
        self.player_segments = {}
        
    def load_and_explore_data(self, file_path):
        """Load dataset and perform initial exploration"""
        print("🎮 Loading Player Behavior Dataset...")
        
        # Load data
        self.data = pd.read_csv(file_path)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Features: {self.data.columns.tolist()}")
        
        # Display basic info
        print("\n📊 Dataset Overview:")
        print(self.data.info())
        
        print("\n📈 Statistical Summary:")
        print(self.data.describe())
        
        # Check for missing values
        print("\n🔍 Missing Values:")
        missing_data = self.data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        return self.data.head()
    
    def feature_engineering(self):
        """Create advanced features for churn prediction"""
        print("\n🔧 Engineering Features for Churn Prediction...")
        
        # Assuming the dataset has these key columns (adjust based on actual dataset)
        # You'll need to modify these based on the actual column names in the dataset
        
        # Example feature engineering (modify based on actual dataset structure)
        if 'SessionsPerWeek' in self.data.columns:
            # Engagement Score
            self.data['EngagementScore'] = (
                self.data['SessionsPerWeek'] * 0.3 +
                self.data.get('AvgSessionDurationMinutes', 0) * 0.4 +
                self.data.get('AchievementsUnlocked', 0) * 0.3
            )
        
        # Activity Decline Indicator
        if 'RecentActivityLevel' in self.data.columns and 'HistoricalActivityLevel' in self.data.columns:
            self.data['ActivityDecline'] = (
                self.data['HistoricalActivityLevel'] - self.data['RecentActivityLevel']
            ) / self.data['HistoricalActivityLevel']
        
        # Monetary Value Features
        if 'InGamePurchases' in self.data.columns:
            self.data['MonetaryValue'] = self.data['InGamePurchases']
            self.data['HighValuePlayer'] = (self.data['MonetaryValue'] > 
                                          self.data['MonetaryValue'].quantile(0.8)).astype(int)
        
        # Recency Features
        if 'DaysSinceLastLogin' in self.data.columns:
            self.data['RecentPlayer'] = (self.data['DaysSinceLastLogin'] <= 7).astype(int)
            self.data['ChurnRisk_Days'] = pd.cut(self.data['DaysSinceLastLogin'], 
                                               bins=[0, 3, 7, 14, 30, 999], 
                                               labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        print(f"✅ Feature Engineering Complete. Dataset now has {self.data.shape[1]} features")
        
    def create_player_segments(self):
        """Create player segments based on behavior patterns"""
        print("\n👥 Creating Player Segments...")
        print(f"Available columns: {self.data.columns.tolist()}")
        
        # Try to identify columns from the actual dataset
        column_mapping = {}
        
        # Look for recency-related columns
        recency_candidates = [col for col in self.data.columns if any(keyword in col.lower() 
                             for keyword in ['day', 'last', 'recent', 'login', 'time'])]
        
        # Look for frequency-related columns  
        frequency_candidates = [col for col in self.data.columns if any(keyword in col.lower() 
                               for keyword in ['session', 'play', 'game', 'freq', 'count', 'times'])]
        
        # Look for monetary-related columns
        monetary_candidates = [col for col in self.data.columns if any(keyword in col.lower() 
                              for keyword in ['purchase', 'spend', 'money', 'revenue', 'pay', 'cost'])]
        
        print(f"Recency candidates: {recency_candidates}")
        print(f"Frequency candidates: {frequency_candidates}")
        print(f"Monetary candidates: {monetary_candidates}")
        
        # Try different approaches based on available data
        try:
            # Method 1: Use the best available columns
            if recency_candidates:
                recency_col = recency_candidates[0]
            elif 'Age' in self.data.columns:
                recency_col = 'Age'  # Use as proxy
            else:
                recency_col = None
                
            if frequency_candidates:
                frequency_col = frequency_candidates[0]
            elif any(col for col in self.data.columns if 'level' in col.lower()):
                frequency_col = [col for col in self.data.columns if 'level' in col.lower()][0]
            else:
                frequency_col = None
                
            if monetary_candidates:
                monetary_col = monetary_candidates[0]
            elif any(col for col in self.data.columns if 'score' in col.lower()):
                monetary_col = [col for col in self.data.columns if 'score' in col.lower()][0]
            else:
                monetary_col = None
            
            # Create segments if we have at least 2 valid columns
            valid_cols = [col for col in [recency_col, frequency_col, monetary_col] if col is not None]
            
            if len(valid_cols) >= 2:
                print(f"Using columns for segmentation: {valid_cols}")
                
                # Create simple scoring system
                for i, col in enumerate(valid_cols):
                    try:
                        # Handle different data types
                        if self.data[col].dtype in ['object', 'category']:
                            # For categorical data, use label encoding
                            le = LabelEncoder()
                            self.data[f'{col}_Score'] = le.fit_transform(self.data[col].astype(str))
                        else:
                            # For numeric data, use quantile-based scoring
                            self.data[f'{col}_Score'] = pd.qcut(self.data[col], 4, labels=[1,2,3,4], duplicates='drop')
                    except Exception as e:
                        print(f"Error processing {col}: {e}")
                        # Fallback: use raw values normalized
                        self.data[f'{col}_Score'] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min()) * 4 + 1
                
                # Create combined score
                score_cols = [f'{col}_Score' for col in valid_cols]
                self.data['Combined_Score'] = self.data[score_cols].mean(axis=1)
                
                # Create segments based on combined score
                self.data['Player_Segment'] = pd.cut(self.data['Combined_Score'], 
                                                   bins=[0, 1.5, 2.5, 3.5, 5], 
                                                   labels=['At_Risk', 'Casual', 'Engaged', 'Champions'])
                
                # Store segment analysis
                self.player_segments = self.data.groupby('Player_Segment').size().to_frame('Count')
                self.player_segments['Percentage'] = (self.player_segments['Count'] / len(self.data) * 100).round(1)
                
                print("✅ Player Segments Created:")
                print(self.player_segments)
                
            else:
                print("⚠️  Not enough suitable columns for segmentation")
                # Create simple random segments for demonstration
                self.data['Player_Segment'] = np.random.choice(['Casual', 'Engaged', 'At_Risk', 'Champions'], 
                                                             size=len(self.data), 
                                                             p=[0.4, 0.3, 0.2, 0.1])
                print("Created random segments for demonstration")
                
        except Exception as e:
            print(f"⚠️  Segmentation failed: {e}")
            # Fallback: create simple segments
            self.data['Player_Segment'] = 'Mixed_Players'
            print("Using single segment as fallback")
    def prepare_data_for_modeling(self, target_column='PlayerChurn'):
        """Prepare data for machine learning models"""
        print(f"\n🎯 Preparing Data for Modeling (Target: {target_column})...")
        
        # Handle missing target column
        if target_column not in self.data.columns:
            # Create synthetic churn labels if not available
            # This is for demonstration - in real scenario, you'd have actual churn data
            print("⚠️  Target column not found. Creating synthetic churn labels...")
            
            # Create churn based on engagement patterns
            churn_conditions = []
            if 'DaysSinceLastLogin' in self.data.columns:
                churn_conditions.append(self.data['DaysSinceLastLogin'] > 14)
            if 'SessionsPerWeek' in self.data.columns:
                churn_conditions.append(self.data['SessionsPerWeek'] < 2)
            
            if churn_conditions:
                self.data[target_column] = np.where(
                    np.logical_or.reduce(churn_conditions), 1, 0
                )
            else:
                # Random churn for demonstration
                self.data[target_column] = np.random.binomial(1, 0.2, size=len(self.data))
        
        # Select features for modeling
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        feature_cols = [col for col in numeric_features 
                       if col not in [target_column, 'PlayerID', 'ID', 'UserID']]
        
        X = self.data[feature_cols]
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        print(f"✅ Data Prepared:")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        print(f"   Churn rate: {y.mean():.2%}")
        
        return feature_cols
    
    def train_models(self):
        """Train multiple ML models for churn prediction"""
        print("\n🤖 Training Machine Learning Models...")
        
        # Handle class imbalance
        if SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        else:
            # Alternative: Manual oversampling of minority class
            X_combined = pd.concat([self.X_train, self.y_train], axis=1)
            majority = X_combined[X_combined.iloc[:, -1] == 0]
            minority = X_combined[X_combined.iloc[:, -1] == 1]
            
            # Oversample minority class
            minority_oversampled = resample(minority, 
                                          replace=True,
                                          n_samples=len(majority),
                                          random_state=42)
            
            # Combine majority class with oversampled minority class
            balanced = pd.concat([majority, minority_oversampled])
            X_train_balanced = balanced.iloc[:, :-1]
            y_train_balanced = balanced.iloc[:, -1]
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_balanced, y_train_balanced)
        self.models['Logistic_Regression'] = lr_model
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=10
        )
        rf_model.fit(X_train_balanced, y_train_balanced)
        self.models['Random_Forest'] = rf_model
        
        # 3. XGBoost (if available)
        if XGB_AVAILABLE:
            print("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_balanced, y_train_balanced)
            self.models['XGBoost'] = xgb_model
        else:
            print("XGBoost not available, training additional Random Forest with different parameters...")
            rf_model2 = RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                max_depth=15,
                min_samples_split=5
            )
            rf_model2.fit(X_train_balanced, y_train_balanced)
            self.models['Random_Forest_V2'] = rf_model2
        
        print("✅ All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📊 Evaluating Model Performance...")
        
        results = {}
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'AUC-ROC': auc_score,
                'Classification_Report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"\n🎯 {name} Results:")
            print(f"   AUC-ROC: {auc_score:.4f}")
            print(f"   Accuracy: {results[name]['Classification_Report']['accuracy']:.4f}")
            print(f"   Precision: {results[name]['Classification_Report']['1']['precision']:.4f}")
            print(f"   Recall: {results[name]['Classification_Report']['1']['recall']:.4f}")
            print(f"   F1-Score: {results[name]['Classification_Report']['1']['f1-score']:.4f}")
        
        return results
    
    def feature_importance_analysis(self):
        """Analyze feature importance from best performing model"""
        print("\n🔍 Analyzing Feature Importance...")
        
        # Use Random Forest for feature importance
        if 'Random_Forest' in self.models:
            rf_model = self.models['Random_Forest']
            feature_names = self.X_train.columns
            
            # Get feature importance
            importance = rf_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("🏆 Top 10 Most Important Features:")
            print(feature_importance_df.head(10))
            
            # Store for visualization
            self.feature_importance['Random_Forest'] = feature_importance_df
            
            return feature_importance_df
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        if 'PlayerChurn' in self.data.columns:
            churn_counts = self.data['PlayerChurn'].value_counts()
            axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%')
            axes[0, 0].set_title('Player Churn Distribution')
        else:
            axes[0, 0].text(0.5, 0.5, 'Churn Data\nNot Available', ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Player Churn Distribution')
        
        # 2. Feature Importance (if available)
        if 'Random_Forest' in self.feature_importance and not self.feature_importance['Random_Forest'].empty:
            top_features = self.feature_importance['Random_Forest'].head(8)
            axes[0, 1].barh(top_features['Feature'], top_features['Importance'])
            axes[0, 1].set_title('Top 8 Feature Importance')
            axes[0, 1].set_xlabel('Importance Score')
        else:
            axes[0, 1].text(0.5, 0.5, 'Feature Importance\nNot Available', ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Feature Importance')
        
        # 3. Player Segments Distribution
        if 'Player_Segment' in self.data.columns:
            segment_counts = self.data['Player_Segment'].value_counts()
            # Ensure we have data to plot
            if len(segment_counts) > 0:
                bars = axes[1, 0].bar(segment_counts.index, segment_counts.values)
                axes[1, 0].set_title('Player Segments Distribution')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Segments\nCreated', ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('Player Segments Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'Segment Data\nNot Available', ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Player Segments Distribution')
        
        # 4. Model Performance Comparison
        if self.models:
            model_names = list(self.models.keys())
            auc_scores = []
            
            for name, model in self.models.items():
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    auc_score = roc_auc_score(self.y_test, y_pred_proba)
                    auc_scores.append(auc_score)
                except:
                    auc_scores.append(0.5)  # Default score if prediction fails
            
            bars = axes[1, 1].bar(model_names, auc_scores)
            axes[1, 1].set_title('Model Performance (AUC-ROC)')
            axes[1, 1].set_ylabel('AUC-ROC Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, auc_scores):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'Model Performance\nNot Available', ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Model Performance')
        
        plt.tight_layout()
        plt.savefig('player_churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved as 'player_churn_analysis_dashboard.png'")
        
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n💡 Generating Business Insights...")
        
        insights = {
            'churn_rate': self.data['PlayerChurn'].mean(),
            'high_risk_players': len(self.data[self.data['PlayerChurn'] == 1]),
            'total_players': len(self.data)
        }
        
        print("🎯 Key Business Insights:")
        print(f"   Overall Churn Rate: {insights['churn_rate']:.2%}")
        print(f"   High-Risk Players: {insights['high_risk_players']:,}")
        print(f"   Total Players Analyzed: {insights['total_players']:,}")
        
        # Segment-based insights
        if 'Player_Segment' in self.data.columns:
            segment_churn = self.data.groupby('Player_Segment')['PlayerChurn'].agg(['mean', 'count'])
            print(f"\n📊 Churn Rate by Player Segment:")
            print(segment_churn)
        
        # Feature-based insights
        if 'Random_Forest' in self.feature_importance:
            top_feature = self.feature_importance['Random_Forest'].iloc[0]['Feature']
            print(f"\n🔍 Most Important Churn Indicator: {top_feature}")
        
        # Retention strategies
        print(f"\n🎯 Recommended Retention Strategies:")
        print("   1. Target high-risk segments with personalized campaigns")
        print("   2. Implement early warning system based on key features")
        print("   3. Create re-engagement programs for at-risk players")
        print("   4. Focus on improving top churn indicators")
        
        return insights
    
    def run_complete_analysis(self, file_path):
        """Run the complete churn analysis pipeline"""
        print("🚀 Starting Complete Player Churn Analysis Pipeline...")
        print("=" * 60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data(file_path)
        
        # Step 2: Feature engineering
        self.feature_engineering()
        
        # Step 3: Create player segments
        self.create_player_segments()
        
        # Step 4: Prepare data for modeling
        feature_cols = self.prepare_data_for_modeling()
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Evaluate models
        results = self.evaluate_models()
        
        # Step 7: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 8: Create visualizations
        self.create_visualizations()
        
        # Step 9: Generate business insights
        insights = self.generate_business_insights()
        
        print("\n" + "=" * 60)
        print("✅ Complete Analysis Pipeline Finished!")
        print("📊 Check 'player_churn_analysis_dashboard.png' for visualizations")
        print("🎯 Ready for Rockstar Analytics portfolio presentation!")
        
        return {
            'results': results,
            'insights': insights,
            'feature_importance': self.feature_importance,
            'models': self.models
        }

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the analysis
    churn_analysis = PlayerChurnAnalysis()
    
    # For demonstration, let's create a sample dataset
    # In real scenario, you'd load from: 'data/predict-online-gaming-behavior-dataset.csv'
    
    print("🎮 Creating Sample Gaming Dataset for Demonstration...")
    
    # Create sample data (remove this when using real dataset)
    np.random.seed(42)
    n_players = 10000
    
    sample_data = pd.DataFrame({
        'PlayerID': range(1, n_players + 1),
        'DaysSinceLastLogin': np.random.exponential(5, n_players),
        'SessionsPerWeek': np.random.poisson(8, n_players),
        'AvgSessionDurationMinutes': np.random.gamma(2, 15, n_players),
        'AchievementsUnlocked': np.random.poisson(25, n_players),
        'InGamePurchases': np.random.exponential(20, n_players),
        'PlayerLevel': np.random.randint(1, 101, n_players),
        'FriendsCount': np.random.poisson(12, n_players),
        'RecentActivityLevel': np.random.uniform(0, 1, n_players),
        'HistoricalActivityLevel': np.random.uniform(0, 1, n_players),
    })
    
    # Save sample data
    sample_data.to_csv('sample_gaming_data.csv', index=False)
    print("✅ Sample dataset created: 'sample_gaming_data.csv'")
    
    # Run the complete analysis
    results = churn_analysis.run_complete_analysis('sample_gaming_data.csv')
    
    print("\n" + "🎯" * 20)
    print("PROJECT READY FOR GITHUB AND RESUME!")
    