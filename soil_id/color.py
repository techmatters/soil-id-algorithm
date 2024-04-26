import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def lab2munsell(color_ref, LAB_ref, LAB):
    """
    Converts LAB color values to Munsell notation using the closest match from a reference
    dataframe.

    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe with LAB and Munsell values.
    - LAB_ref (list): Reference LAB values.
    - LAB (list): LAB values to be converted.

    Returns:
    - str: Munsell color notation.
    """
    idx = pd.DataFrame(euclidean_distances([LAB], LAB_ref)).idxmin(axis=1).iloc[0]
    munsell_color = (
        f"{color_ref.at[idx, 'hue']} "
        f"{int(color_ref.at[idx, 'value'])}/{int(color_ref.at[idx, 'chroma'])}"
    )
    return munsell_color


def munsell2rgb(color_ref, munsell_ref, munsell):
    """
    Converts Munsell notation to RGB values using a reference dataframe.

    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe with Munsell and RGB values.
    - munsell_ref (pd.DataFrame): Reference dataframe with Munsell values.
    - munsell (list): Munsell values [hue, value, chroma] to be converted.

    Returns:
    - list: RGB values.
    """
    idx = munsell_ref.query(
        f'hue == "{munsell[0]}" & value == {int(munsell[1])} & chroma == {int(munsell[2])}'
    ).index[0]
    return [color_ref.at[idx, col] for col in ["r", "g", "b"]]


def rgb2lab(color_ref, rgb_ref, rgb):
    """
    Convert RGB values to LAB color values using a reference dataframe.

    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe containing RGB and LAB values.
    - rgb_ref (list): Reference RGB values.
    - rgb (list): RGB values to be converted.

    Returns:
    - list: LAB values.
    """
    idx = pd.DataFrame(euclidean_distances([rgb], rgb_ref)).idxmin(axis=1).iloc[0]
    return [color_ref.at[idx, col] for col in ["L", "A", "B"]]
