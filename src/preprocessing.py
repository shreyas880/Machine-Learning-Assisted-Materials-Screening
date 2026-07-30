import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from utils import *
from load_material_profiles import load_datasets

FEATURE_COLUMNS = [
    "density",
    "volume",
    "nsites",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_metal",
    "is_stable",
]

TARGET_COLUMNS = [
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "thermal_conductivity_clarke",
    "thermal_conductivity_cahill",
    "debye_temperature",
    "homogeneous_poisson"
]

BOOLEAN_COLUMNS = ['is_metal', 'is_stable']

sets = ["elements", "binary", "ternary", "quaternary", "low_density", "medium_density", "high_density", "highly_stable", "metastable", "metals", "nonmetals"]

dict_columns = ["bulk_modulus", "shear_modulus", "thermal_conductivity"]

def expand_composition(composition):
    element_map = {
        element: 0
        for element in ALL_ELEMENTS
    }

    for element, value in composition.items():
        element_map[element] = value

    return pd.Series(element_map)

datasets = load_datasets(*sets)

all_materials = combine_dataframes(*datasets.values())
ORIGINAL_SIZE = len(all_materials)

os.makedirs("plots/dataset", exist_ok=True)

def remove_target_outliers(df, columns):

    for column in columns:

        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        print("Remaining rows :", len(df))
    return df

expanded_materials = all_materials

for column in dict_columns:
    expanded_materials = expand_dictionary_column(expanded_materials, column)

# Removing the columns where one or more targets are missing
before_cleaning = expanded_materials.copy()
expanded_materials = expanded_materials.dropna(subset=TARGET_COLUMNS)
AFTER_MISSING_VALUES = len(expanded_materials)

# Removing outliers from target columns
expanded_materials = remove_target_outliers(expanded_materials, TARGET_COLUMNS)
AFTER_OUTLIERS = len(expanded_materials)

after_cleaning = (expanded_materials.copy())

# Preserve material meta data
METADATA_COLUMNS = [
    "material_id",
    "formula_pretty",
    "composition_reduced",
    "density",
    "volume",
    "nsites",
    "is_metal",
    "is_stable"
]

metadata = expanded_materials[METADATA_COLUMNS].copy()

ALL_ELEMENTS = (expanded_materials["elements"].explode().unique())
composition_features = (expanded_materials["composition_reduced"].apply(expand_composition))

expanded_materials['is_metal'] = (expanded_materials['is_metal'].fillna(False))
expanded_materials[BOOLEAN_COLUMNS] = (expanded_materials[BOOLEAN_COLUMNS].astype(int))

X = pd.concat([expanded_materials[FEATURE_COLUMNS], composition_features], axis=1)

y = expanded_materials[TARGET_COLUMNS]

# Creating train test splits
X_complete = X
y_complete = y


X_train, X_temp, y_train, y_temp, metadata_train, metadata_temp = train_test_split(
    X_complete,
    y_complete,
    metadata,
    test_size=0.2,
    random_state=42,
    shuffle=True

)

X_val, X_test, y_val, y_test, metadata_val, metadata_test = train_test_split(
    X_temp,
    y_temp,
    metadata_temp,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

TRAIN_SIZE = len(X_train)
VALIDATION_SIZE = len(X_val)
TEST_SIZE = len(X_test)

# Z-score scaling for features
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_val = feature_scaler.transform(X_val)
X_test = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train)

y_val = target_scaler.transform(y_val)

y_test = target_scaler.transform(y_test)

# Exporting useful data
def preprocess():

    return(

        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        target_scaler,
        feature_scaler,

        metadata_test
    )


#===============================================================
# DATASET EVOLUTION
#===============================================================

def plot_dataset_evolution():

    labels = [
        "Original\nDataset",
        "Missing Values\nRemoved",
        "Outliers\nRemoved"
    ]


    sizes = [
        ORIGINAL_SIZE,
        AFTER_MISSING_VALUES,
        AFTER_OUTLIERS
    ]


    plt.figure(figsize=(8,5))


    bars = plt.bar(labels, sizes)

    plt.ylabel("Number of Materials")


    plt.title("Dataset Evolution During Preprocessing")


    percentages = [
        100,
        round((AFTER_MISSING_VALUES/ORIGINAL_SIZE)*100, 2),
        round((AFTER_OUTLIERS/ORIGINAL_SIZE)*100, 2)
    ]

    for bar,size,percentage in zip(bars, sizes, percentages):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            size,
            f"{size}\n({percentage}%)",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(
        "plots/dataset/dataset_evolution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


#===============================================================
# TARGET DISTRIBUTION COMPARISON
#===============================================================

def plot_distribution_comparison():

    fig,axes = plt.subplots(2, 6, figsize=(18,6))


    for index,column in enumerate(TARGET_COLUMNS):
        axes[0,index].hist(
            before_cleaning[column]
            .dropna(),
            bins=30
        )

        axes[0,index].set_title(
            column,
            fontsize=8
        )

        axes[1,index].hist(
            after_cleaning[column],
            bins=30
        )

    axes[0,0].set_ylabel("Before\nCleaning")

    axes[1,0].set_ylabel("After\nCleaning")

    plt.suptitle("Target Distribution Comparison", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        "plots/dataset/target_distribution_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_dataset_split():
    labels = [
        "Train",
        "Validation",
        "Test"
    ]
    sizes = [
        TRAIN_SIZE,
        VALIDATION_SIZE,
        TEST_SIZE
    ]


    percentages = [
        round((TRAIN_SIZE/sum(sizes))*100,2),
        round((VALIDATION_SIZE/sum(sizes))*100,2),
        round((TEST_SIZE/sum(sizes))*100,2)
    ]

    plt.figure(
        figsize=(7,5)
    )

    bars = plt.bar(
        labels,
        sizes
    )

    for bar,size,percentage in zip(
        bars,
        sizes,
        percentages
    ):

        plt.text(
            bar.get_x()+bar.get_width()/2,
            size,
            f"{size}\n({percentage}%)",
            ha="center",
            va="bottom"
        )
    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Train Validation Test Split"
    )

    plt.tight_layout()
    plt.savefig(
        "plots/dataset/train_validation_test_split.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def generate_dataset_plots():
    plot_dataset_evolution()
    plot_distribution_comparison()
    plot_dataset_split()

    
#===============================================================
# MAIN
#===============================================================

if __name__ == "__main__":

    generate_dataset_plots()