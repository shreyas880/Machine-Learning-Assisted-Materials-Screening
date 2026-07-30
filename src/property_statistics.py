import pandas as pd
import numpy as np

def property_statistics(df):

    statistics = {}

    total = len(df)

    for column in df.columns:

        missing = df[column].isna().sum()
        populated = total-missing
        
        percentage = round(populated/total*100,2)
        statistics[column]={
            "total":total,
            "missing":missing,
            "populated":populated,
            "percentage_populated":percentage
        }

    return pd.DataFrame(statistics).T