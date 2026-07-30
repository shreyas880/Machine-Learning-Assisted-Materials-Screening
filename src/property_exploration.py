from utils import *
from property_statistics import property_statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

print('started')
SEARCHES = [
    ("elements",{
        "num_elements":(1,1)
    }),

    ("binary",{
        "num_elements":(2,2)
    }),

    ("ternary",{
        "num_elements":(3,3)
    }),

    ("quaternary",{
        "num_elements":(4,4)
    }),

    ("low_density",{
        "density":(0,3)
    }),

    ("medium_density",{
        "density":(3, 8)
    }),

    ("high_density",{
        "density":(8, 30)
    }),

    ("highly_stable",{
        "energy_above_hull": (0,0.01)
    }),

    ("metastable",{
        "energy_above_hull": (0.01,0.1)
    }),

    ("metals",{
        "is_metal":True
    }),

    ("nonmetals",{
        "is_metal":False
    }),
]

datasets = {}
start = time.time()

for name,kwargs in SEARCHES:

    materials = get_materials(**kwargs)

    save_dataset(name, materials)

    datasets[name] = materials

end = time.time()

print(f"Total time: {end-start}")