import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_csv(list_files): #OK
    """Importing a list of csv files into a pandas Dataframe for further explorations, correcting mirror effect

    Parameters
    ----------
    list_file :  String list
        A list of csv file's name containing all driving informations for multiple users

    Returns
    ---------
    dataframes : Dataframe list
        Converted csv files

    """
    dataframes = []

    for file in list_files:
        tmp = pd.read_csv(file)
        tmp['position_y'] *= -1  # AirSim y axis is inverted compared to matplotlib's
        dataframes.append(tmp)

    return dataframes
