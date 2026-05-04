# HomeFix Experiential Analytics Challenge

> First-place winner. Judged by HomeFix's Director of IT.
> Montclair State University, MS Business Analytics.

A machine learning system that predicts financing turn-downs before sales reps leave the office, and identifies upsell opportunities inside approved customers. Built for HomeFix Custom Remodeling.

---

## The Problem

HomeFix runs a high-cost field sales model. Every lead means a sales rep drives to a customer's home, pitches services, and submits a financing application. All of that happens before anyone knows if the customer will qualify.

The numbers we found:

- 74% of financing applications were turned down
- 3 out of 4 sales visits generated zero revenue
- No system existed to predict turn-downs in advance

The business question: how do we stop spending money on the wrong customers and find more revenue inside the right ones?

---

## The Data

We worked with five datasets totalling over 142,000 records:

| Dataset | Records | Source |
|---|---|---|
| Finance Applications | 3,397 | HomeFix CRM |
| Soft Pull Credit Runs | 81,922 | HomeFix CRM |
| ZIP-Level Median Income | 30,411 | US Census ACS |
| ZIP-Level Home Values | 26,300 | Zillow ZHVI |
| Appointments and Leads | 30,000+ | HomeFix CRM |

The HomeFix data is confidential and is not included in this repository. The external Census and Zillow data is publicly available.

---

## The Approach

**1. Defined the target variable**
Approved and Auto Approved decisions = 1. All other statuses (Counter Offered, Auto Declined, Declined, Withdrawn, Pending) = 0. Modeled only finalized decision records, not pipeline.

**2. Engineered 12 features**
Combined credit tier, requested amount, downpayment, lender, product, and engineered ratios like income-to-request and downpayment percentage.

**3. Integrated external data**
Merged US Census income and Zillow home values at the ZIP level. Both became the top signals in the final model.

**4. Time-based train and test split**
Train: October 2024 to October 2025 (2,717 records). Test: October 2025 to January 2026 (680 records). Oversampled the minority class to handle the 74% imbalance.

---

## The Models

### Model 1: Turn-Down Risk Classifier

A Gradient Boosting Classifier that predicts the probability of turn-down before any visit is scheduled. Each lead receives a risk score from 0 to 1.

**Test set results:**
- Flagged 320 out of 490 likely turn-downs
- Correctly retained 83.68% of approved customers
- Top features: median income (ZIP), product type, home value (ZIP), plan requested

### Model 2: Approved Amount Predictor

A Gradient Boosting Regressor that predicts how much each approved customer is actually eligible to borrow. Surfaces customers who qualify for more than they originally requested.

**Test set results:**

| Metric | Baseline | Our Model | Improvement |
|---|---|---|---|
| MAPE | 88.48% | 14.01% | 74.47 pp |
| MAE | $12,775 | $2,496 | $10,279 |
| R² | -0.008 | 0.93 | 0.94 |

---

## Business Impact

| Metric | Value |
|---|---|
| Top-line revenue preserved | $2.7M |
| Untapped financing capacity identified | $913K |
| Net economic impact | $168K |
| Customers flagged for upsell | 178 |

What the system enables in practice:

- **Smarter dispatch**: Sales reps see a risk score before driving to a home
- **Targeted upsell**: High-headroom customers flagged for larger financing offers
- **Better lead buying**: Marketing budget shifts toward high-income, high-value ZIPs

---

## Tech Stack

- **Language**: Python 3.12
- **Libraries**: pandas, numpy, scikit-learn, openpyxl
- **Models**: HistGradientBoostingClassifier, GradientBoostingRegressor, LogisticRegression
- **Tools**: Google Colab, Jupyter, Git

---

## Repository Structure

```
homefix-analytics-challenge/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── hcr_analysis.py
├── notebooks/
│   └── hcr_analysis.ipynb
├── presentation/
│   ├── HomeFix_Project_Recap.pptx
│   └── HomeFix_Project_Recap.pdf
└── data/
    └── README.md   (data not included, confidential)
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/homefix-analytics-challenge.git
cd homefix-analytics-challenge

# Install dependencies
pip install -r requirements.txt

# Place the HomeFix data files in the data/ folder
# (data is not included for confidentiality)

# Run the analysis
python src/hcr_analysis.py
```

---

## Key Takeaways

The biggest lesson had nothing to do with the models. Companies care about profit, not methodology. If you cannot connect your analysis to a dollar figure, you have not finished the job.

Three things mattered most:

1. External data beats internal data when used right. ZIP-level income and home values outranked credit tier as predictors.
2. The class imbalance was the real challenge. Oversampling the minority class made everything else possible.
3. Threshold optimization is a business decision, not a technical one. We built the tradeoff table so HomeFix can pick the cutpoint that matches their cost structure.

---

## Author

**Aric Mahmood**
MS Business Analytics, Montclair State University
[LinkedIn](https://www.linkedin.com/in/yourprofile) | aricm1@montclair.edu

---

## Acknowledgments

Thanks to HomeFix Custom Remodeling for providing the dataset and to their Director of IT for judging the competition. Thanks to Montclair State University's Feliciano School of Business for organizing the challenge.
