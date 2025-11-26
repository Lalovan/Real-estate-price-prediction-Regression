import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer, StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor

filename = "cleaned_properties.csv"
df = pd.read_csv(filename)

X = df.drop(columns = ["price","id","zip_code","latitude","longitude"])
y = df["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state = 36)

# Set of features
numeric_features = ["cadastral_income","primary_energy_consumption_sqm","nbr_bedrooms","nbr_frontages","total_area_sqm"]
skewed_features = ["total_area_sqm"]
categorical_onehot = ["heating_type","equipped_kitchen", "epc"]
categorical_target = ["subproperty_type","province"]
binary_features = ["fl_terrace", "fl_garden", "fl_swimming_pool", "fl_furnished"]

# Function that helps getting back the features names
def get_column_names(ct):
    """
    Return list of output column names produced by a fitted ColumnTransformer `ct`.
    Handles Pipelines, SimpleImputer(add_indicator=True) inside pipelines,
    and transformers that implement get_feature_names_out.
    """
    feature_names = []

    for name, transformer, cols in ct.transformers_:
        # Skip dropped transformers
        if transformer == 'drop':
            continue

        # passthrough: keep original names
        if transformer == 'passthrough':
            feature_names.extend(list(cols))
            continue

        # Some ColumnTransformer entries may be (name, transformer, slice) where
        # transformer is a Pipeline or transformer instance.
        # We'll treat Pipeline specially.
        if isinstance(transformer, Pipeline):
            
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                try:
                    names = last_step.get_feature_names_out(cols)
                    feature_names.extend(list(names))
                    continue
                except Exception:
                    # if it fails for any reason, fall through to other checks
                    pass

            imputer_with_indicator = None
            for step_name, step_obj in transformer.steps:
                if isinstance(step_obj, SimpleImputer) and getattr(step_obj, "add_indicator", False):
                    imputer_with_indicator = step_obj
                    break

            if imputer_with_indicator is not None:
                # Imputer keeps original number of columns + indicator cols (one per input col with NaNs seen during fit)
                feature_names.extend(list(cols))
                if hasattr(imputer_with_indicator, 'indicator_'):
                    indicator_names = [f"{cols[i]}_missing_flag" for i in imputer_with_indicator.indicator_.features_]
                    feature_names.extend(indicator_names)
                continue

            feature_names.extend(list(cols))
            continue

        # If transformer is not a Pipeline
        # Try to use get_feature_names_out if present
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                names = transformer.get_feature_names_out(cols)
                feature_names.extend(list(names))
                continue
            except Exception:
                pass

        # Check if this transformer itself is a SimpleImputer with add_indicator=True
        if isinstance(transformer, SimpleImputer) and getattr(transformer, "add_indicator", False):
            feature_names.extend(list(cols))
            if hasattr(transformer, 'indicator_'): # Thie priece resolves the issue when missing_fl colummn is created but not needed, causing issue when converting to df
                indicator_names = [f"{cols[i]}_missing_flag" for i in transformer.indicator_.features_]
                feature_names.extend(indicator_names)
            continue

        # final fallback: original column names
        feature_names.extend(list(cols))

    return feature_names

# Preporcessor Pipeline

# Pipeline for numeric columns (imputation, scale, capping (capping needs to come as a parameter from train data - leakage issue))
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)),
])

# No log-transformation is done; as in ct we cannot pass both lists of vars, separate pipeline is indicated
skew_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)),
])

onehot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant')), 
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline for label/ordinal categorical features - "MISSING" is treated as a category
target_pipeline = Pipeline([
    ('target_enc', TargetEncoder(smoothing=1.0))
])

preprocessor_forest_boost = ColumnTransformer([('num', numeric_pipeline, [f for f in numeric_features if f not in skewed_features]),
    ('skewed',skew_pipeline, skewed_features),
    ('onehot', onehot_pipeline, categorical_onehot),
    ('target', target_pipeline, categorical_target),
    ('binary', 'passthrough', binary_features) # Just passing them as-is
])

xgboost_model = Pipeline(steps=[
    ("preprocess", preprocessor_forest_boost),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9
    ))
])

xgboost_model.fit(X_train,y_train)

y_pred = xgboost_model.predict(X_test)

# Evaluating festure importance by weight, cover and gain
preprocessor = xgboost_model.named_steps["preprocess"]
feature_names = get_column_names(preprocessor)

booster = xgboost_model.named_steps["model"].get_booster() # This is where the booster is stored

importance_gain = booster.get_score(importance_type='gain')

df_gain = (
    pd.DataFrame(list(importance_gain.items()), columns=['Feature', 'Gain'])
    .sort_values('Gain', ascending=False)
)

# Extracting importance metrics
importance_gain = booster.get_score(importance_type='gain')
importance_weight = booster.get_score(importance_type='weight')
importance_cover = booster.get_score(importance_type='cover')

# Map "f0", "f1" etc to actual feature names
importance_gain_named = {feature_names[int(k[1:])]: v for k, v in importance_gain.items()}
importance_weight_named = {feature_names[int(k[1:])]: v for k, v in importance_weight.items()}
importance_cover_named = {feature_names[int(k[1:])]: v for k, v in importance_cover.items()}

all_features = feature_names

df_importance = pd.DataFrame({
    'Feature': all_features,
    'Gain': [importance_gain_named.get(f, 0) for f in all_features],
    'Weight': [importance_weight_named.get(f, 0) for f in all_features],
    'Cover': [importance_cover_named.get(f, 0) for f in all_features]
})

# Optional: sort by Gain descending
df_importance = df_importance.sort_values(by='Gain', ascending=False)

df_importance

# Evaluation of the metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    results = {
        'R2_train': r2_score(y_train, y_pred_train),
        'R2_test': r2_score(y_test, y_pred_test),
        'MAE_train': mean_absolute_error(y_train, y_pred_train),
        'MAE_test': mean_absolute_error(y_test, y_pred_test),
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    
    return results

results_xgb = evaluate_model(xgboost_model, X_train, y_train, X_test, y_test)

df_results = pd.DataFrame([results_xgb],index=['XGBoost'])
df_results

#Cross-validation
pipeline = xgboost_model #xgboost_model
cv_r2 = cross_val_score(pipeline, X, y, scoring='r2', cv=10)
rmse_scores = - cross_val_score(pipeline, X,y, scoring='neg_root_mean_squared_error', cv=10)

print("CV R-sqr per fold: ", cv_r2)
print("Mean CV R-sqr per fold: ", cv_r2.mean())
print("RMSE(each fold): ", rmse_scores)
print("RMSE mean: ", rmse_scores.mean() )

# Saving the model

# Saving the expected data types to be passed to the model as dummies/new data

joblib.dump(pipeline, "model.pkl")

print("Pipeline saved as model.pkl")