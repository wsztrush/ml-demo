import pandas as pd
import numpy as np

index = pd.date_range('10/1/1999', periods=1100)
ts = pd.Series(np.random.normal(0.5, 2, 1100), index)
ts = ts.rolling(window=100, min_periods=100).mean().dropna()

key = lambda x: x.year
zscore = lambda x : (x - x.mean()) / x.std()

result = ts.groupby(key).transform(zscore)

print(result)
