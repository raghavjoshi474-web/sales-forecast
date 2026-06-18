# Sales Forecasting & Product Recommendation

An end-to-end data science project combining time series sales forecasting with a collaborative filtering recommendation engine, built on real retail datasets.

## Problem Statement
Retail businesses need to predict future sales for inventory planning, and recommend relevant products to customers to increase basket size. This project addresses both challenges.

## Datasets
- **Forecasting:** [Walmart Sales Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset) — 6,435 weekly sales records across 45 stores (2010-2012)
- **Recommendation:** [Online Retail Dataset - Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data) — 397,924 genuine purchase transactions across 4,339 customers and 3,665 products

## Part 1 — Sales Forecasting

### What I Did
- Performed time series EDA — discovered strong holiday season spikes every Nov-Dec
- Confirmed economic indicators (Temperature, Fuel Price, CPI) have weak correlation with sales
- Engineered lag features: previous week's sales and 4-week moving average per store
- Built a Random Forest Regressor with chronological train/test split (no data leakage)
- Diagnosed why a naive ML approach failed and fixed it with feature engineering

### Results
| Approach | MAE |
|---|---|
| Baseline (4-week moving average) | $84,201 |
| Random Forest (no lag features) | $427,094 ❌ |
| Random Forest (with lag features) | $47,026 ✅ |

**Reduced forecast error (MAE) by 44.15% compared to baseline moving average**

### Key Finding
Recent sales history (Avg_Last_4_Weeks, Prev_Week_Sales) dominates all other features in importance — far more predictive than economic indicators or calendar features alone.

## Part 2 — Product Recommendation Engine

### What I Did
- Cleaned transaction data — removed missing CustomerIDs and cancelled orders
- Built a customer-product purchase matrix (4,339 × 3,877)
- Applied cosine similarity to find products with similar purchase patterns
- Built a recommendation function returning top-N similar products for any input product

### Example Output
Input: WHITE HANGING HEART T-LIGHT HOLDER
Recommendations:
- RED HANGING HEART T-LIGHT HOLDER (0.66 similarity)
- GIN + TONIC DIET METAL SIGN (0.75 similarity)
- WASHROOM METAL SIGN (0.64 similarity)

## Technologies Used
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Cosine Similarity
