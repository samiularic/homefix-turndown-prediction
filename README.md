# homefix-turndown-prediction
Gradient Boosting models predicting financing turn-down probability and approved loan amounts for HomeFix Custom Remodeling. First place winner, Montclair State University Analytics Competition.

HomeFix Experiential Analytics Challenge, First Place Winner
A graduate business case competition project built at Montclair State University, Feliciano School of Business, MS Business Analytics program, Spring 2026.
This project delivers a two-model machine learning system that predicts financing turn-down probability and approved loan amounts for HomeFix Custom Remodeling, a real East Coast exterior remodeling company. The system was built on real company data and won first place among all competing teams, judged by HomeFix's Director of IT.

The Problem
HomeFix runs a high-cost field sales model. Every lead requires a sales rep to physically drive to a customer's home before any financing decision is made. 74% of financing applications were being turned down, meaning 3 out of 4 visits generated zero revenue. No predictive system existed to identify unqualified leads before dispatch.

What It Does
On submission of a financing application, the system:

Scores each lead with a turn-down risk probability from 0 to 1 before any sales visit is scheduled
Predicts the actual approved loan amount for qualified customers, surfacing upsell opportunities for customers eligible to borrow more than they requested
Flags high-headroom customers for targeted upsell outreach
Enables marketing budget allocation toward high-income, high-value ZIP codes


How It Was Built
The team merged internal HomeFix application and credit data with external US Census ACS ZIP-level income records and Zillow ZIP-level home value records. Both external sources ranked as stronger predictive signals than internal credit tier data.
A time-based train and test split was applied (October 2024 to October 2025 for training, October 2025 to January 2026 for testing) to simulate real deployment conditions. Class imbalance from the 74% turn-down rate was handled through oversampling of the minority class. Twelve features were engineered from credit, lender, product, and ratio variables.
Two separate models were trained and evaluated independently.

Architecture
Raw Data Sources
      |
      |-- HomeFix financing applications (3,397 records)
      |-- Soft pull credit records (81,922 records)
      |-- US Census ACS ZIP income data (30,411 records)
      |-- Zillow ZIP home value data (26,300 records)
      |
      v
Feature Engineering and External Data Merge (ZIP-level join)
      |
      v
Time-Based Train / Test Split + Oversampling
      |
      |-- Model 1: HistGradientBoostingClassifier
      |   Turn-down probability score (0 to 1) per lead
      |
      |-- Model 2: GradientBoostingRegressor
          Predicted approved loan amount per qualified customer

Model Performance
Turn-Down Classifier

Flagged 320 out of 490 likely turn-downs in the test set
Correctly retained 83.68% of approved customers

Approved Amount Regressor

R-squared: 0.93
Mean absolute error reduced from $12,775 to $2,496 per customer
14% MAPE vs 88% baseline, a 74 percentage point improvement


Business Impact

$2.7M in top-line revenue preserved by correctly retaining approved customers
$913K in untapped financing capacity identified across 178 customers for upsell
$168K total net economic impact from avoided wasted visits and new upsell revenue
Enabled smarter dispatch, targeted upsell, and ZIP-level marketing reallocation


Tech Stack
LayerToolData wrangling and feature engineeringPython 3.12, pandas, NumPyMachine learning modelsscikit-learn (HistGradientBoostingClassifier, GradientBoostingRegressor, LogisticRegression)Excel integrationopenpyxlDevelopment environmentGoogle Colab, Jupyter NotebookVersion controlGit, GitHubPresentationMicrosoft PowerPoint, Canva

Outcome
First place winner among all competing MS Business Analytics teams at Montclair State University. Judged by HomeFix's Director of IT.
