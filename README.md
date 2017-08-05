# toady

### Easily visualize high-dimensional data in 2d space.

<img src="https://image.ibb.co/jp1PMa/common_toad_2382962_640.jpg"/>

### Basic Usage

```python 
import pandas as pd
from toady.toady import toady

data = pd.read_csv('data.csv')
X = data['feature_1', 'feature_2', 'feature_3']
y = data['target']

toady(X, y)
```

### Parameters

- **X** : { pd.DataFrame }
    
    DataFrame containing input data / predictors / features.
    (Can contain categorical, missing data, etc)

- **y** : { pd.Series }
    
    Series containing the target variable.
    (Can be categorical)
    
- **point_labels** : { list }
    
    List containing the label of each scatter plot point.

- **impute_model** : { type }, default 'sklearn.preprocessing.Imputer'
    
    Model used for imputing missing values in the data. 

- **impute_params** : { dict }, default empty dict

    Params fed to impute_model.

- **scale_model** : { type }, default 'sklearn.preprocessing.RobustScaler'

    Model used for scaling the data.

- **scale_params** : { dict }, default empty dict

    Params fed to scale_params.

- **embed_model** : { type }, default 'sklearn.manifold.Isomap'

    Model used for embedding the data onto 2d space.

- **embed_params** : { dict }, default empty dict
    
    Params fed to embed_model.

- **scatter_params** : { dict }, default empty dict

    Params fed to the scatter plot.

- **css** : { CSS string }, default seen in README

    CSS string for tooltips.

- **verbose** : { bool }, default False

    Whether or not informative messages are shown at each step of the toady process.
