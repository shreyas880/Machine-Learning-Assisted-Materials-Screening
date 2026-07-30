import numpy as np
import pandas as pd
from load_material_profiles import load_datasets
from property_statistics import property_statistics
from utils import *

sets = ["elements", "binary", "ternary", "quaternary", "low_density", "medium_density", "high_density", "highly_stable", "metastable", "metals", "nonmetals"]

datasets = load_datasets(*sets)

all_materials = combine_dataframes(*datasets.values())
print(property_statistics(all_materials))