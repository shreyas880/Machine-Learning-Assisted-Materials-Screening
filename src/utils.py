import pandas as pd
import numpy as np
from mp_api.client import MPRester
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math

LIMIT=10000

fields_map = {
    "summary": ["material_id", "formula_pretty", "elements", "composition_reduced", "density", "volume", "is_metal", "homogeneous_poisson"],
    "elasticity": ["material_id", "thermal_conductivity", "youngs_modulus", "bulk_modulus", "shear_modulus", "debye_temperature", "composition_reduced"],
    "thermo": ["material_id", "volume", "formation_energy_per_atom", "energy_above_hull", "nsites", "is_stable"]
}

MAT_PROJ_API_KEY='materials_project_api_key_here'


# --------------------------------------------------------------------------------------------------
#                                               CONSTANTS
# --------------------------------------------------------------------------------------------------

TARGET_COLUMNS = [
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "thermal_conductivity_clarke",
    "thermal_conductivity_cahill",
    "debye_temperature",
    "homogeneous_poisson"
]

FEATURE_COLUMNS = [
    "density",
    "volume",
    "nsites",
    "formation_energy_per_atom",
    "energy_above_hull"
]


TARGET_COLUMNS = [
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "thermal_conductivity_clarke",
    "thermal_conductivity_cahill",
    "debye_temperature",
    "homogeneous_poisson"
]

def batch_ids(ids, batch_size=500):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i+batch_size]

def get_materials(**kwargs):
    ids = get_material_ids(**kwargs)

    print(f"\nFound {len(ids)} material ids.\n")

    # Multithreaded parallel processing
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # One worker for each of the three endpoints

        summary_future = executor.submit(
            get_properties,
            domain="summary",
            material_ids=ids,
            fields=fields_map["summary"]
        )

        elasticity_future = executor.submit(
            get_properties,
            domain="elasticity",
            material_ids=ids,
            fields=fields_map["elasticity"]
        )

        thermo_future = executor.submit(
            get_properties,
            domain="thermo",
            material_ids=ids,
            fields=fields_map["thermo"]
        )

        summary_properties = summary_future.result()

        elasticity_properties = elasticity_future.result()

        thermo_properties = thermo_future.result()

    combined = combine_properties(
        summary_properties,
        elasticity_properties,
        thermo_properties
    )

    print(f"\nFinal materials : {len(combined)}\n")

    return combined

def get_material_ids(limit=None, **kwargs):
    with MPRester(MAT_PROJ_API_KEY) as mpr:
        # unable to use the limit as a search query for some reason, idk?
        material_ids = mpr.materials.summary.search(
            fields=["material_id"],
            **kwargs
        )
        material_ids = [material_id["material_id"] for material_id in material_ids]


        material_ids = material_ids[:limit] if (limit is not None and len(material_ids) > limit) else material_ids

        return material_ids

def get_properties(domain, material_ids, batch_size=500, **kwargs):
    materials = []
    
    total_batches = math.ceil(len(material_ids)/batch_size)

    with MPRester(MAT_PROJ_API_KEY) as mpr:
        endpoint = getattr(mpr.materials, domain)

        print(f"\n{domain.upper()}")
        print("-"*50)

        # To have a sense of how far we've come in the dataset collection, im using tqdm
        for batch in tqdm(
            batch_ids(material_ids, batch_size),
            total=total_batches,
            desc=f"{domain}"
        ):

            result = endpoint.search(
                material_ids=batch,
                **kwargs
            )

            materials.extend(result)

    print(f"\nRetrieved {len(materials)} materials.")
        
    return clean_properties(materials)

def clean_properties(materials):
    # The materials still contain two metadata fields that are not useful and i wanted to remove them  
    cleaned = {}
    
    for item in materials:
        cleaned[item.material_id] = (
            item.model_dump(
            exclude={
                "fields_not_requested",
                "unavailable_fields"
                }
            )
        )

    return cleaned

def combine_properties(*datasets):

    common_ids = set(datasets[0].keys())

    for dataset in datasets[1:]:
        common_ids &= set(dataset.keys())


    combined = {}

    for m_id in common_ids:

        combined[m_id] = {}

        combined[m_id].update(dataset[m_id] for dataset in datasets) 


    return pd.DataFrame(list(combined.values()))

def combine_dataframes(*datasets):
    unique_materials = {}
    for dataset in datasets:
        for material in dataset.to_dict(
            orient="records"
        ):
            m_id = material["material_id"]
            unique_materials[m_id] = material

    return pd.DataFrame(list(unique_materials.values()))

def combine_materials(*datasets):

    combined = pd.concat(datasets, ignore_index=True)

    combined = combined.drop_duplicates(subset='material_id')

    return combined

def save_dataset(name,materials):
    df = pd.DataFrame(materials)
    df.to_pickle(f"datasets/{name}.pkl")

def expand_dictionary_column(df, column):
    expanded = (df[column].apply(pd.Series).add_prefix(f"{column}_"))

    df = pd.concat([df, expanded], axis=1)

    return df