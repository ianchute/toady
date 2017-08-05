# toady

### Easily visualize high-dimensional data in 2d space.

<img src="https://image.ibb.co/jp1PMa/common_toad_2382962_640.jpg"/>

### Basic Usage

Below is a very simple example using the Iris dataset.

```python 
import pandas as pd
from toady import toady

data = pd.read_csv('iris.csv')

features = ['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']
X = data[features]
y = data['Species']

toady(X, y)
```

<img src="https://image.ibb.co/gCwYga/sample0.png">

### Example with point labels

Below map 7 features of the world's top universities onto 2d space and coloring the points based on it's score. We also add a label for each point:

```python
data = pd.read_csv('cwurData.csv')

features = [
    'quality_of_education',
    'alumni_employment',
    'quality_of_faculty',
    'publications',
    'influence',
    'citations',
    'patents',
]

X = data[features]
y = data['score']
labels = list(data['institution'].values + ' (' + data['year'].apply(str).values + ')')

toady(X, y, labels)
```

In this embedding, the *very top* schools in the world (e.g. Harvard, Princeton, etc.) are actually near each other in our embedding (hover on points to reveal labels):

<img src="https://image.ibb.co/eWjrZv/sample1.png">

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
