import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def lab2munsell(color_ref, LAB_ref, lab):
    """
    Converts LAB color values to Munsell notation using the closest match from a reference
    dataframe.

    Parameters:
    - lab (list): LAB values to be converted.

    Returns:
    - str: Munsell color notation.
    """
    idx = pd.DataFrame(euclidean_distances([lab], LAB_ref)).idxmin(axis=1).iloc[0]
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


