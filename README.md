# Player-Churn-Prediction-Behavioral-Segmentation

🎮 Player Churn Prediction & Behavioral Segmentation
This end-to-end analytics project simulates the work of Rockstar’s analytics team by building a robust pipeline to analyze player behavior, segment users, and predict churn using real-world-style gaming data. The workflow integrates machine learning, behavioral feature engineering, and business insight generation to enable data-driven decisions for player retention and engagement strategies.

📌 Project Highlights
📦 Dataset: 40,034 players × 16 behavioral attributes (sessions, purchases, achievements, recency, playtime, etc.)
🔧 Feature Engineering: Engagement score, activity decline, monetary value, recency buckets, and churn risk levels


👥 Player Segmentation:
At_Risk: 5,999 players | Churn Rate: 57.9%
Casual: 20,551 players | Churn Rate: 56.9%
Engaged: 10,477 players | Churn Rate: 57.4%
Champions: 2,990 players | Churn Rate: 55.6%

🚀 ML Pipeline Performance
Models Trained: Logistic Regression, Random Forest, XGBoost
Top Accuracy: 77.4%
AUC-ROC: 0.84 (Random Forest)
Top Feature: DaysSinceLastLogin emerged as the strongest predictor of churn

Output: All visual insights captured in player_churn_analysis_dashboard.png

📊 Key Insights
📉 Churn Rate: 57.14% overall (22,875 churned out of 40,034)

🎯 Most Important Indicator: DaysSinceLastLogin
📊 Segment-Wise Churn Patterns show similar rates across tiers, emphasizing the need for tailored retention per segment


💡 Retention Strategy Recommendations
Target high-risk player segments (At_Risk & Casual) with personalized engagement campaigns
Implement early churn detection based on activity drop-offs and recency
Launch reactivation programs for players showing behavioral decline
Focus design/testing on features most predictive of disengagement

🧠 Tech Stack
Python: pandas, scikit-learn, XGBoost, matplotlib, seaborn

ML Tools: SMOTE, GridSearchCV, feature importance (SHAP fallback)

Deliverables: Custom ML class, reusable notebook, dashboard PNG, business insights

![image](https://github.com/user-attachments/assets/311ecba5-96f3-4c88-a6a2-757c7783ac2e)
