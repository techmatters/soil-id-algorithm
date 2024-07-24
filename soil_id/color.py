import numpy as np
import pandas as pd
import skimage
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
    return [color_ref.at[idx, col] for col in ["srgb_r", "srgb_g", "srgb_b"]]


def convert_rgb_to_lab(row):
    """
    Converts RGB values to LAB.
    """
    if pd.isnull(row["srgb_r"]) or pd.isnull(row["srgb_g"]) or pd.isnull(row["srgb_b"]):
        return np.nan, np.nan, np.nan

    result = skimage.color.rgb2lab([row["srgb_r"], row["srgb_g"], row["srgb_b"]])

    return result


def getProfileLAB(data_osd, color_ref):
    """
    The function processes the given data_osd DataFrame and computes LAB values for soil profiles.
    """
    # Convert the specific columns to numeric
    data_osd[["top", "bottom", "srgb_r", "srgb_g", "srgb_b"]] = data_osd[
        ["top", "bottom", "srgb_r", "srgb_g", "srgb_b"]
    ].apply(pd.to_numeric)

    if not validate_color_data(data_osd):
        return pd.DataFrame(
            np.nan, index=np.arange(200), columns=["cielab_l", "cielab_a", "cielab_b"]
        )

    data_osd = correct_color_depth_discrepancies(data_osd)

    data_osd["cielab_l"], data_osd["cielab_a"], data_osd["cielab_b"] = zip(
        *data_osd.apply(lambda row: convert_rgb_to_lab(row), axis=1)
    )

    l_intpl, a_intpl, b_intpl = [], [], []

    for index, row in data_osd.iterrows():
        l_intpl.extend([row["cielab_l"]] * (int(row["bottom"]) - int(row["top"])))
        a_intpl.extend([row["cielab_a"]] * (int(row["bottom"]) - int(row["top"])))
        b_intpl.extend([row["cielab_b"]] * (int(row["bottom"]) - int(row["top"])))

    lab_intpl = pd.DataFrame({"cielab_l": l_intpl, "cielab_a": a_intpl, "cielab_b": b_intpl}).head(
        200
    )
    return lab_intpl


def validate_color_data(data):
    """
    Validates color data based on given conditions.
    """
    if data.top.isnull().any() or data.bottom.isnull().any():
        return False
    if data.srgb_r.isnull().all() or data.srgb_g.isnull().all() or data.srgb_b.isnull().all():
        return False
    if data.top.iloc[0] != 0:
        return False
    return True


def correct_color_depth_discrepancies(data):
    """
    Corrects depth discrepancies by adding layers when needed.
    """
    layers_to_add = []
    for i in range(len(data.top) - 1):
        if data.top.iloc[i + 1] > data.bottom.iloc[i]:
            layer_add = pd.DataFrame(
                {
                    "top": data.bottom.iloc[i],
                    "bottom": data.top.iloc[i + 1],
                    "srgb_r": np.nan,
                    "srgb_g": np.nan,
                    "srgb_b": np.nan,
                },
                index=[i + 0.5],
            )
            layers_to_add.append(layer_add)

    if layers_to_add:
        data = pd.concat([data] + layers_to_add).sort_index().reset_index(drop=True)

    return data


def calculate_deltaE2000(LAB1, LAB2):
    """
    Computes the Delta E 2000 value between two LAB color values.

    Args:
        LAB1 (list): First LAB color value.
        LAB2 (list): Second LAB color value.

    Returns:
        float: Delta E 2000 value.
    """

    L1star, a1star, b1star = LAB1
    L2star, a2star, b2star = LAB2

    C1abstar = math.sqrt(a1star**2 + b1star**2)
    C2abstar = math.sqrt(a2star**2 + b2star**2)
    Cabstarbar = (C1abstar + C2abstar) / 2.0

    G = 0.5 * (1.0 - math.sqrt(Cabstarbar**7 / (Cabstarbar**7 + 25**7)))

    a1prim = (1.0 + G) * a1star
    a2prim = (1.0 + G) * a2star

    C1prim = math.sqrt(a1prim**2 + b1star**2)
    C2prim = math.sqrt(a2prim**2 + b2star**2)

    h1prim = math.atan2(b1star, a1prim) if (b1star != 0 or a1prim != 0) else 0
    h2prim = math.atan2(b2star, a2prim) if (b2star != 0 or a2prim != 0) else 0

    deltaLprim = L2star - L1star
    deltaCprim = C2prim - C1prim

    if (C1prim * C2prim) == 0:
        deltahprim = 0
    elif abs(h2prim - h1prim) <= 180:
        deltahprim = h2prim - h1prim
    elif abs(h2prim - h1prim) > 180 and (h2prim - h1prim) < 360:
        deltahprim = h2prim - h1prim - 360.0
    else:
        deltahprim = h2prim - h1prim + 360.0

    deltaHprim = 2 * math.sqrt(C1prim * C2prim) * math.sin(deltahprim / 2.0)

    Lprimbar = (L1star + L2star) / 2.0
    Cprimbar = (C1prim + C2prim) / 2.0

    if abs(h1prim - h2prim) <= 180:
        hprimbar = (h1prim + h2prim) / 2.0
    elif abs(h1prim - h2prim) > 180 and (h1prim + h2prim) < 360:
        hprimbar = (h1prim + h2prim + 360) / 2.0
    else:
        hprimbar = (h1prim + h2prim - 360) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(hprimbar - 30.0)
        + 0.24 * math.cos(2.0 * hprimbar)
        + 0.32 * math.cos(3.0 * hprimbar + 6.0)
        - 0.20 * math.cos(4.0 * hprimbar - 63.0)
    )

    deltatheta = 30.0 * math.exp(-(math.pow((hprimbar - 275.0) / 25.0, 2.0)))
    RC = 2.0 * math.sqrt(Cprimbar**7 / (Cprimbar**7 + 25**7))
    SL = 1.0 + (0.015 * (Lprimbar - 50.0) ** 2) / math.sqrt(20.0 + (Lprimbar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cprimbar
    SH = 1.0 + 0.015 * Cprimbar * T
    RT = -math.sin(2.0 * deltatheta) * RC

    kL, kC, kH = 1.0, 1.0, 1.0
    term1 = (deltaLprim / (kL * SL)) ** 2
    term2 = (deltaCprim / (kC * SC)) ** 2
    term3 = (deltaHprim / (kH * SH)) ** 2
    term4 = RT * (deltaCprim / (kC * SC)) * (deltaHprim / (kH * SH))

    return math.sqrt(term1 + term2 + term3 + term4)


# Not currently implemented for US SoilID
def interpolate_color_values(top, bottom, color_values):
    """
    Interpolates the color values based on depth.

    Args:
        top (pd.Series): Top depths.
        bottom (pd.Series): Bottom depths.
        color_values (pd.Series): Corresponding color values.

    Returns:
        np.array: Interpolated color values for each depth.
    """

    if top[0] != 0:
        raise ValueError("The top depth must start from 0.")

    MisHrz = any([top[i + 1] != bottom[i] for i in range(len(top) - 1)])
    if MisHrz:
        raise ValueError("There is a mismatch in horizon depths.")

    color_intpl = []
    for i, color_val in enumerate(color_values):
        color_intpl.extend([color_val] * (bottom[i] - top[i]))

    return np.array(color_intpl)


# Not currently implemented for US SoilID
def getColor_deltaE2000_OSD_pedon(data_osd, data_pedon):
    """
    Calculate the Delta E 2000 value between averaged LAB values of OSD and pedon samples.

    The function interpolates the color values based on depth for both OSD and pedon samples.
    It then computes the average LAB color value for the 31-37 cm depth range.
    Finally, it calculates the Delta E 2000 value between the two averaged LAB values.

    Args:
        data_osd (object): Contains depth and RGB data for the OSD sample.
            - top: List of top depths.
            - bottom: List of bottom depths.
            - r, g, b: Lists of RGB color values corresponding to each depth.

        data_pedon (object): Contains depth and LAB data for the pedon sample.
            - [0]: List of bottom depths.
            - [1]: DataFrame with LAB color values corresponding to each depth.

    Returns:
        float: Delta E 2000 value between the averaged LAB values of OSD and pedon.
        Returns NaN if the data is not adequate for calculations.
    """
    # Extract relevant data for OSD and pedon
    top, bottom, r, g, b = (
        data_osd.top,
        data_osd.bottom,
        data_osd.r,
        data_osd.g,
        data_osd.b,
    )
    ref_top, ref_bottom, ref_lab = (
        [0] + data_pedon[0][:-1],
        data_pedon[0],
        data_pedon[1],
    )

    # Convert RGB values to LAB for OSD
    osd_colors_rgb = interpolate_color_values(top, bottom, list(zip(r, g, b)))
    osd_colors_lab = [skimage.color.rgb2lab([[color_val]])[0][0] for color_val in osd_colors_rgb]

    # Calculate average LAB for OSD at 31-37 cm depth
    osd_avg_lab = np.mean(osd_colors_lab[31:37], axis=0) if len(osd_colors_lab) > 31 else np.nan
    if np.isnan(osd_avg_lab).any():
        return np.nan

    # Convert depth values to LAB for pedon
    pedon_colors_lab = interpolate_color_values(
        ref_top,
        ref_bottom,
        list(zip(ref_lab.iloc[:, 0], ref_lab.iloc[:, 1], ref_lab.iloc[:, 2])),
    )

    # Calculate average LAB for pedon at 31-37 cm depth
    pedon_avg_lab = (
        np.mean(pedon_colors_lab[31:37], axis=0) if len(pedon_colors_lab) > 31 else np.nan
    )
    if np.isnan(pedon_avg_lab).any():
        return np.nan

    # Return the Delta E 2000 value between the averaged LAB values
    return calculate_deltaE2000(osd_avg_lab, pedon_avg_lab)
