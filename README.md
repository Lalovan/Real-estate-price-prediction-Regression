# Real-estate-price-prediction-Regression

### ⏩ At a Glance: ###

Building a Machine Learning (ML) model for price prediction based on scraped, cleaned and analysed real estate data of the Belgian market.

# 🚀 Project mission #

This repository documents the ML stage of Immo Eliza’s broader data pipeline, whose mission is to deliver faster and more accurate property price estimations across Belgium. 

Following the initial data scraping and exploratory data analysis (EDA), this stage focuses on preprocessing, model development, and evaluation for predictive model. Linear (Linear Regression) and Tree-Based models (Random Forest, XGBoost) have been tested in the process. 

# 📌 Project Context #
Immo Eliza (imaginary real estate firm) aims to strengthen decision-making and valuation precision by integrating data-driven modeling. After gathering Belgian real estate data (web scraping) and conducting early analysis (DEA), the next step is turning these insights into a robust predictive model.

This repository provides:
- A complete preprocessing pipeline;
- 3 regression models (linear & tree-based);
- Performance evaluation and comparison;
- Saved, reusable models for future prediction workflows;

# 🧭 Workflow Overview #
1. Data Cleaning

The dataset (scraped earlier in the project) is prepared through:
- Removing duplicates;
- Fixing obvious inconsistencies;
- Handling missing values;
- Dropping irrelevant or unusable columns;
2. Preprocessing Pipeline

A fully reusable pipeline is constructed to ensure consistency between training and future prediction:
- Imputation: Numerical features: median (LR), and -1 + missing_fl (tree-based)
- Encoding: OneHotEncoder and TargetEncoder
- Log-transformation of the target and a set of other numerical, highly-skewed features
- Feature Scaling
- Standardization (z-score scaling)
- Correlation filtering
- Removal of highly collinear features
- Capping of data at 1% and 99% (outliers strategy)

All transformations are kept inside a Pipeline/ColumnTransformer so they can be applied identically to new data.

3. Model Training

Three models are trained and compared:
- Linear Regression (baseline)
- Random Forest
- XGBoost

Training is performed only on the training split to avoid leakage.

4. Model Evaluation

Models are evaluated using:
- R² Score
- Mean Absolute Error (MAE)
- RMSE

Additional checks:
- Underfitting/overfitting behavior

5. Model Saving

Trained models are serialized using pickle and joblib, enabling later use in:
- A predict.py script
- API deployment
- Further product development

# 🧪 Results (Summary) #

| Model | R² (Train) | R² (Test) | MAE (Train) | MAE (Test) | RMSE (Train) | RMSE (Test) |
|-------|-----------:|----------:|-------------:|------------:|--------------:|-------------:|
| Linear Regression | 0.33 | 0.32 | 129401 | 131042 | 354869 | 368252 | 
| Random Forest     | 0.91 | 0.62 | 41770 | 104093 | 127739 | 276016 | 
| XGBoost           | 0.82 | 0.61 | 88724 | 110107 | 181186 | 277788 | 
| XGBoost (+ Cross-validation)   | - | 0.59 (vs Test) | - | - | 280961 | - | 


The baseline linear regression provides a useful reference point. However, it does not fully capture non-linear interactions in the data. Moreover, a major limitaiton of this model is the sensitivity to outliers. 

Tree-based models show significant improvements in predictive accuracy, with XGBoost achieving the best balance between bias and variance across train and test data (p.p. gaps between R2s). Moreover, cross-validation technique helped descreasing this gap further.

This is the reason why XGBoost has been selected as a prediction basis. The final model is saved for deployment and integrated into the prediction script (`predict.py`).

# 🌳 Repository Structure #

## 📂 Repository Structure

```
Real-estate-price-prediction-Regression
│
├── 01_Models.ipynb        # Notebook for preprocessing, modeling & evaluation
├── train.py               # Training script (pipeline + models + saving)
├── predict.py             # Generate predictions using the saved model
│
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore rules
├── README.md              # Project documentation

```



# ⚙️ Installation and Execution #

**1. Clone the repository:**

`git clone https://github.com/Lalovan/Real-estate-price-prediction-Regression.git`


**2. Create and activate a virtual environment (optional):**

`python -m venv venv`

`source venv/bin/activate      # macOS/Linux`

`venv\Scripts\activate         # Windows`


**3. Install dependencies:**

`pip install -r requirements.txt`

**4. Train a model**

`python scripts/train.py`

**5. Predict the price of a new property**

`python scripts/predict.py`


# ⏰ Timeline #

4 working days

# 🔮 Next Steps #
- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Expand the feature set with external datasets (geospatial, socioeconomic, etc.)
- Deploy the model behind an API endpoint
- Build a small web UI for end-users
- Continuous integration and automated model tests