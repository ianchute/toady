import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

from sklearn.preprocessing import RobustScaler, Imputer
from sklearn.manifold import Isomap

DEFAULT_CSS = """
text.mpld3-text, div.mpld3-tooltip {
  font-family: Arial, Helvetica, sans-serif;
  font-weight: bold;
  color: #39FF14;
  text-shadow: #000 0px 0px 3px,   #000 0px 0px 3px,   #000 0px 0px 3px,
             #000 0px 0px 3px,   #000 0px 0px 3px,   #000 0px 0px 3px;
  opacity: 1.0;
  border: 0px;}
"""

def _to_numpy(X):
    categorical_cols = X.dtypes[X.dtypes == 'object'].keys()
    for col in categorical_cols:
        encoded = pd.get_dummies(X[col], prefix=col)
        X = X.drop(col, axis=1)
        X = pd.concat([X, encoded], axis=1)
    return X.values.astype(np.float64)

def _apply_model(X, model_f, params):
    model = model_f()
    for attr in params.keys():
        if hasattr(model, attr):
            setattr(model, attr, params[attr])
    return model.fit_transform(X)

def _scatter_plot(X, y, point_labels, scatter_params, css):
    X = pd.DataFrame(X, columns=['x','y'])
    fig, ax = plt.subplots(figsize=(8,8))

    if 'figsize' in scatter_params:
        fig, ax = plt.subplots(figsize=scatter_params['figsize'])
        del scatter_params['figsize']

    colors, classes = None, None

    if y.dtype == 'object':
        colors, classes = y.factorize()
    else:
        colors = y

    scatter = ax.scatter(
        x=X['x'], 
        y=X['y'],
        c=colors,
        **scatter_params
    )

    if point_labels is not None and len(point_labels) > 0:
        tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=point_labels, css=css)
        mpld3.plugins.connect(fig, tooltip)
    else:
        mpld3.plugins.connect(fig)

    return mpld3.display()


def toady(X, y, point_labels=[], \
    impute_model=Imputer, impute_params={}, \
    scale_model=RobustScaler, scale_params={}, \
    embed_model=Isomap, embed_params={ 'n_jobs': -1 }, \
    scatter_params={}, css=DEFAULT_CSS, \
    verbose=False \
):
    """Easily visualize high-dimensional data in 2d space.
    Parameters
    ----------
    X : { pd.DataFrame }
        DataFrame containing input data / predictors / features.
        (Can contain categorical, missing data, etc)
    y : { pd.Series }
        Series containing the target variable.
        (Can be categorical)
    point_labels : { list }
        List containing the label of each scatter plot point.
    impute_model : { type }, default 'sklearn.preprocessing.Imputer'
        Model used for imputing missing values in the data. 
    impute_params : { dict }, default empty dict
        Params fed to impute_model.
    scale_model : { type }, default 'sklearn.preprocessing.RobustScaler'
        Model used for scaling the data.
    scale_params : { dict }, default empty dict
        Params fed to scale_params.
    embed_model : { type }, default 'sklearn.manifold.Isomap'
        Model used for embedding the data onto 2d space.
    embed_params : { dict }, default empty dict
        Params fed to embed_model.
    scatter_params : { dict }, default empty dict
        Params fed to the scatter plot.
    css : { CSS string }, default seen in README
        CSS string for tooltips.
    verbose : { bool }, default False
        Whether or not informative messages are shown at each step of the toady process."""
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be a Pandas DataFrame (pd.DataFrame)!')
    if not isinstance(y, pd.Series):
        raise Exception('y must be a Pandas Series (pd.Series)!')
    if point_labels is not None \
        and not (isinstance(point_labels, list) and all(map(lambda label: isinstance(label, str), point_labels))):
        raise Exception('point_labels must be a list of strings or None!')
    _models = [impute_model, scale_model, embed_model]
    if not all(map(lambda model: isinstance(model, type), _models)):
        raise Exception('All models must be types!')
    _params = [impute_params, scale_params, embed_params]
    if not all(map(lambda p: isinstance(p, dict), _params)):
        raise Exception('All params must be dicts!')
    if not isinstance(css, str):
        raise Exception('css must be a string!')
    if not isinstance(verbose, bool):
        raise Exception('verbose must be a bool!')

    embed_params['n_components'] = 2 # This is always 2 for now.

    if verbose:
        print('Converting data to numpy array...')
    X = _to_numpy(X)

    if verbose:
        print('Imputing missing values (if any)...')
    X = _apply_model(X, impute_model, impute_params)
    
    if verbose:
        print('Scaling...')
    X = _apply_model(X, scale_model, scale_params)

    if verbose:
        print('Embedding data into two dimensions...')
    X = _apply_model(X, embed_model, embed_params)

    if verbose:
        print('Creating scatter plot...')
    return _scatter_plot(X, y, point_labels, scatter_params, css)
