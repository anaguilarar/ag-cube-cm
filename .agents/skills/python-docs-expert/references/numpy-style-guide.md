# NumPy Style Docstring Guide

This guide describes how to write Python docstrings using the **NumPy Style**, which is widely used in scientific and data-oriented Python projects (like those using `numpy`, `pandas`, `scipy`, etc.).

## Standard Section Order
1. **Summary**: A brief, one-line summary.
2. **Extended Summary**: (Optional) More details about what the function does.
3. **Parameters**: Describe function arguments.
4. **Returns**: Describe return values.
5. **Yields**: (For generators) Describe yielded values.
6. **Raises**: List exceptions the function might raise.
7. **Examples**: Provide illustrative code examples.

## Parameters Section
Format each parameter as:
`name : type` followed by an indented description.

```python
Parameters
----------
config : OmegaConf
    The configuration object loaded via OmegaConf.
extent : list of float, optional
    Bounding box [xmin, ymin, xmax, ymax]. Defaults to None.
ncores : int, default 1
    Number of CPU cores to use for processing.
```

## Returns Section
Similar to parameters, but if the return is unnamed, just the type is used.

```python
Returns
-------
pd.DataFrame
    A DataFrame containing the processed weather data.
bool
    True if the download was successful, False otherwise.
```

## Raises Section
```python
Raises
------
FileNotFoundError
    If the specified configuration file does not exist.
ValueError
    If the extent is invalid.
```

## Examples Section
Use the standard Python prompt style (`>>>`).

```python
Examples
--------
>>> from spatialdata.gis_functions import get_boundaries_from_path
>>> extent = get_boundaries_from_path("path/to/shapefile.shp")
>>> print(extent)
[-89.5, 13.2, -88.1, 14.5]
```
