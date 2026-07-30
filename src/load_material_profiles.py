import pandas as pd

def load_dataset(name):
    return pd.read_pickle(
        f"datasets/{name}.pkl"
    )

def load_datasets(*names):
    datasets={}
    for name in names:
        datasets[name] = load_dataset(name)

    return datasets