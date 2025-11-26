import pandas as pd
import joblib


model_path = "model.pkl"
pipeline = joblib.load(model_path)

data = pd.DataFrame([{
    "total_area_sqm": 150,
    "cadastral_income": 980,
    "primary_energy_consumption_sqm": 100,
    "nbr_bedrooms": 4,
    "nbr_frontages": 4,
    "subproperty_type": "HOUSE",
    "province": "Antwerp",
    "fl_terrace": 0,
    "fl_garden": 1,
    "fl_swimming_pool": 0,
    "fl_furnished": 1,
    "epc": "good",
    "equipped_kitchen": "MISSING",
    "heating_type": "GAS"
}])

int_cols = ["total_area_sqm", "cadastral_income", "primary_energy_consumption_sqm",
            "nbr_bedrooms", "nbr_frontages", "fl_terrace", "fl_garden", 
            "fl_swimming_pool", "fl_furnished"]
for col in int_cols:
    data[col] = data[col].astype("Int64")

prediction = pipeline.predict(data)
print(prediction)