import pandas as pd
from energy_efficiency.exception import energy_exception
import yaml
import os,sys


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise energy_exception(e,sys) from e
