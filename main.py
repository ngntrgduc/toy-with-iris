import iris
import numpy as np
import pandas as pd

data = iris.load_data()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
print(df)
