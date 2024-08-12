# Online Normalization (onorm)


```python
import numpy as np
from plotnine import ggplot, geom_point, aes, theme_minimal
import pandas as pd
```


```python
n = 100
d = 2

X = np.random.normal(size = (n, d))
df = pd.DataFrame(X, columns=["X1", "X2"])

ggplot(df, aes("X1", "X2")) + geom_point() + theme_minimal()
```


    
![png](README_files/README_2_0.png)
    

