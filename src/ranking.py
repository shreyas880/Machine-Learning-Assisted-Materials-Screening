import numpy as np
import pandas as pd
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from preprocessing import preprocess
from model import TARGET_COLUMNS

os.makedirs("plots/recommendations", exist_ok=True)

def print_title(title):
        print()
        print("="*70)
        print(title)
        print("="*70)
        print()

(X_train, X_val, X_test, y_train, y_val, y_test, target_scaler, feature_scaler, metadata_test) = preprocess()

model = load_model("best_material_model.keras")
y_pred = model.predict(X_test)

y_pred = target_scaler.inverse_transform(y_pred)
y_test = target_scaler.inverse_transform(y_test)

predictions = pd.DataFrame(
    y_pred,
    columns=TARGET_COLUMNS
)

actual_values = pd.DataFrame(
    y_test,
    columns=TARGET_COLUMNS
)

metadata_test = metadata_test.reset_index(drop=True)
predictions = predictions.reset_index(drop=True)
actual_values = actual_values.reset_index(drop=True)

results = pd.concat([metadata_test, predictions], axis=1)
results.to_csv("material_predictions.csv", index=False)

for column in TARGET_COLUMNS:
    results[f"{column}_score"] = (results[column].rank(pct=True)*100)


results["mechanical_score"] = (
    (0.45*results["bulk_modulus_vrh_score"])
    +
    (0.45*results["shear_modulus_vrh_score"])
    +
    (0.10*results["homogeneous_poisson_score"])
)

results["thermal_score"] = (
    (0.40*results["thermal_conductivity_clarke_score"])
    +
    (0.40*results["thermal_conductivity_cahill_score"])
    +
    (0.20*results["debye_temperature_score"])
)

results["overall_score"] = (
    (0.40*results["mechanical_score"])
    +
    (0.60*results["thermal_score"])
)

results["rank"] = (results["overall_score"].rank(ascending=False, method="min"))

results = results.sort_values(
    by="overall_score",
    ascending=False
)

results.to_csv("ranked_materials.csv", index=False)

SUPER_HARD_THRESHOLD = 90
THERMAL_THRESHOLD = 95
BALANCED_THRESHOLD = 85

super_hard_materials = results[(results["mechanical_score"] >= SUPER_HARD_THRESHOLD)]
super_hard_materials = (super_hard_materials.sort_values(by="mechanical_score", ascending=False))

high_thermal_materials = results[(results["thermal_score"] >= THERMAL_THRESHOLD)]
high_thermal_materials = (high_thermal_materials.sort_values(by="thermal_score", ascending=False))

balanced_materials = results[
    (results["mechanical_score"] >= BALANCED_THRESHOLD) &
    (results["thermal_score"] >= BALANCED_THRESHOLD)
]


balanced_materials = (
    balanced_materials
    .sort_values(
        by="overall_score",
        ascending=False
    )
)

super_hard_materials.to_csv(
    "super_hard_materials.csv",
    index=False
)

high_thermal_materials.to_csv(
    "high_thermal_materials.csv",
    index=False
)



balanced_materials.to_csv(
    "balanced_materials.csv",
    index=False
)

print_title('TOP 10 BALANCED MATERIALS')
print(balanced_materials[["material_id", "formula_pretty", "overall_score", "mechanical_score", "thermal_score" ]].head(10))

print_title('TOP 10 SUPER HARD MATERIALS')
print(super_hard_materials[["material_id", "formula_pretty", "mechanical_score"]].head(10))

print_title('TOP 10 HIGH THERMAL MATERIALS')
print(high_thermal_materials[[ "material_id", "formula_pretty", "thermal_score"]].head(10))

print_title("TOP 20 MATERIALS")
print(results[[ "rank", "material_id", "formula_pretty", "overall_score", "mechanical_score", "thermal_score"]].head(20))

print_title("TOP 10 MECHANICAL MATERIALS")
print(results.sort_values(by="mechanical_score", ascending=False),
    [[ "material_id", "formula_pretty", "mechanical_score"]].head(10))

print_title("TOP 10 THERMAL MATERIALS")
print(results.sort_values(by="thermal_score",ascending=False),
      [["material_id", "formula_pretty", "thermal_score"]].head(10))

print_title("MATERIAL PREDICTIONS")
print(results.head())
print_title("TOTAL MATERIALS")
print(len(results))
print_title("AVAILABLE COLUMNS")
print(results.columns)

def generate_material_profiles(df):
    
    print_title("TOP 10 THERMAL MATERIALS")
    print("TOP 5 MATERIAL PROFILES")

    top_5 = df.sort_values(
        by="overall_score",
        ascending=False
    ).head(5)

    for _, row in top_5.iterrows():
        print(f"Material ID : {row['material_id']}")
        print(f"Formula     : {row['formula_pretty']}")
        print("MECHANICAL PROPERTIES")

        print(f"Bulk Modulus Score : "f"{round(row['bulk_modulus_vrh_score'],2)}")

        print(f"Shear Modulus Score : "f"{round(row['shear_modulus_vrh_score'],2)}")

        print(f"Mechanical Score : "f"{round(row['mechanical_score'],2)}")

        print("THERMAL PROPERTIES")
        print(f"Thermal Conductivity (Clarke) Score : "f"{round(row['thermal_conductivity_clarke_score'],2)}")

        print(f"Thermal Conductivity (Cahill) Score : "f"{round(row['thermal_conductivity_cahill_score'],2)}")

        print(f"Debye Temperature Score : "f"{round(row['debye_temperature_score'],2)}")

        print(f"Poisson Ratio Score : "f"{round(row['homogeneous_poisson_score'],2)}")

        print(f"Thermal Score : "f"{round(row['thermal_score'],2)}")

        print("FINAL SCORES")

        print(f"Overall Score : "f"{round(row['overall_score'],2)}")

        print("RECOMMENDATIONS")
        if row["mechanical_score"] > 90:
            print("Suitable for : Super Hard Materials")

        if row["thermal_score"] > 90:
            print("Suitable for : High Thermal Applications")

        if (row["mechanical_score"] > 85 and row["thermal_score"] > 85):
            print("Suitable for : Balanced Applications")

def export_results(results):
    os.makedirs("results",exist_ok=True)
    results.to_csv("results/all_materials.csv",index=False)
    results.sort_values(by="overall_score", ascending=False).head(20).to_csv("results/top_20_materials.csv", index=False)

    results.sort_values(by="mechanical_score", ascending=False).head(10).to_csv("results/top_mechanical.csv", index=False)

    results.sort_values(by="thermal_score", ascending=False).head(10).to_csv("results/top_thermal.csv", index=False)

    balanced = results[(results["mechanical_score"] > 85) & (results["thermal_score"] > 85)]

    balanced.to_csv(
        "results/top_balanced.csv",
        index=False
    )

generate_material_profiles(results)
export_results(results)
