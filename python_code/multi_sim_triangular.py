import numpy as np
import pandas as pd
from numpy.linalg import cholesky
from skbio.stats.composition import ilr_inv
from scipy.stats import spearmanr


def simulate_correlated_triangular(n, params, correlation_matrix):
    """
    Simulate correlated triangular distributed variables.

    Parameters:
    - n: Number of samples.
    - params: List of tuples, where each tuple contains three parameters (a, b, c) for
      the triangular distribution.
    - correlation_matrix: 2D numpy array representing the desired correlations between
      the variables.

    Returns:
    - samples: 2D numpy array with n rows and as many columns as there are sets of
      parameters in params.
    """

    # Generate uncorrelated standard normal variables
    uncorrelated_normal = np.random.normal(size=(n, len(params)))

    # Cholesky decomposition of the correlation matrix
    L = cholesky(correlation_matrix)

    # Compute correlated variables using Cholesky decomposition
    correlated_normal = uncorrelated_normal @ L

    # Transform standard normal variables to match triangular marginal distributions
    samples = np.zeros((n, len(params)))

    for i, (a, b, c) in enumerate(params):
        normal_var = correlated_normal[:, i]
        u = norm.cdf(normal_var)  # Transform to uniform [0, 1] range

        # Transform the uniform values into triangularly distributed values
        condition = u <= (b - a) / (c - a)
        samples[condition, i] = a + np.sqrt(u[condition] * (c - a) * (b - a))
        samples[~condition, i] = c - np.sqrt((1 - u[~condition]) * (c - a) * (c - b))

    return samples


"""
use 'ilr' and 'ilr_inv' functions from the skbio.stats.composition package
"""

# def clr(x):
#     """
#     Compute the centered log-ratio transformation.
#     """
#     gm = np.exp(np.mean(np.log(x)))  # Geometric mean
#     return np.log(x / gm)
#
# def ilr(x, basis=None):
#     """
#     Compute the isometric log-ratio transformation.
#     """
#     if basis is None:
#         # Default to a specific orthonormal basis for simplicity
#         # This will work for compositions of length 3
#         # For other lengths, a basis related to a specific SBP should be provided
#         basis = np.array([
#             [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
#             [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)]
#         ])
#     clr_values = clr(x)
#     return clr_values @ basis.T


def acomp(X, parts=None, total=1):
    if parts is None:
        parts = list(range(X.shape[1]))

    parts = list(set(parts))

    if isinstance(X, pd.DataFrame):
        Xn = X.iloc[:, parts].to_numpy()
    else:
        Xn = X[:, parts]

    Xn /= Xn.sum(axis=1)[:, np.newaxis] / total

    return gsi_simshape(Xn, X)


def gsi_simshape(x, oldx):
    if oldx.ndim >= 2:
        return x
    return x.flatten() if oldx.ndim == 0 else x.reshape(-1)


"""
Modeling Steps:
  Step 1. Calculate a local soil property correlation matrix uisng the representative values
          from SSURGO.

  Step 2. Steps performed on each row:
    a. Simulate sand/silt/clay percentages using the 'simulate_correlated_triangular' function,
       using the global correlation matrix and the local l,r,h values for each particle fraction.
       Format as a composition using 'acomp'
    b. Perform the isometric log-ratio transformation.
    c. Extract l,r,h values (min, median, max for ilr1 and ilr2) and format into a params object
       for simiulation.
    d. Simulate all properties and then permorm inverse transform on ilr1 and ilr2 to obtain sand,
       silt, and clay values.
    e. Append simulated values to dataframe

  Step 3. Run Rosetta and other Van Genuchten equations to calcuate AWS in top 50 cm using
          simulated dataframe.
"""

# Step 1. Calculate a local soil property correlation matrix

# Extract columns with names ending in '_r'
df_r = df[[col for col in df.columns if col.endswith("_r")]]

# Compute the local correlation matrix (Spearman correlation matrix)
selected_columns = df.drop(columns=["sandtotal_r", "silttotal_r", "claytotal_r"])
correlation_matrix, _ = spearmanr(selected_columns, axis=0)

ilr_site_txt = irl(df[["sandtotal_r", "silttotal_r", "claytotal_r"]])

selected_columns["ilr1"] = ilr_site_txt[:, 0]
selected_columns["ilr2"] = ilr_site_txt[:, 1]

correlation_matrix_data = selected_columns[
    [
        "ilr1",
        "ilr2",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "dbthirdbar_r",
        "wthirdbar_r",
        "wfifteenbar_r",
    ]
]
local_correlation_matrix, _ = spearmanr(correlation_matrix_data, axis=0)

# Step 2. Simulate data for each row, with the number of simulations equal to the comppct_r*10

# Global soil texture correlation matrix (used for initial simulation)
texture_correlation_matrix = np.array(
    [
        [1.0000000, -0.76231798, -0.67370589],
        [-0.7623180, 1.00000000, 0.03617498],
        [-0.6737059, 0.03617498, 1.00000000],
    ]
)

results = []
for _, row in df.iterrows():
    # 2a. Simulate sand/silt/clay percentages
    # 1. Extract and format data params
    sand_params = [row["sandtotal_l"], row["sandtotal_r"], row["sandtotal_h"]]
    silt_params = [row["silttotal_l"], row["silttotal_r"], row["silttotal_h"]]
    clay_params = [row["claytotal_l"], row["claytotal_r"], row["claytotal_h"]]

    params_txt = list[sand_params, silt_params, clay_params]

    # 2. Perform processing steps on data
    # Convert simulated data using the acomp function and then compute the
    # isometric log-ratio transformation.
    simulated_txt = acomp(
        simulate_correlated_triangular(
            row["comppct_r"] * 10, params_txt, texture_correlation_matrix
        )
    )
    simulated_txt_ilr = irl(simulated_txt)

    # Extract min, median, and max for the first two ilr transformed columns.
    ilr1_values = simulated_txt_ilr[:, 0]
    ilr2_values = simulated_txt_ilr[:, 1]

    ilr1_l, ilr1_r, ilr1_h = (
        ilr1_values.min(),
        np.median(ilr1_values),
        ilr1_values.max(),
    )
    ilr2_l, ilr2_r, ilr2_h = (
        ilr2_values.min(),
        np.median(ilr2_values),
        ilr2_values.max(),
    )

    # Create the list of parameters.
    params = [
        [ilr1_l, ilr1_r, ilr1_h],
        [ilr2_l, ilr2_r, ilr2_h],
        [row["dbthirdbar_l"], row["dbthirdbar_r"], row["dbthirdbar_h"]],
        [row["wthirdbar_l"], row["wthirdbar_r"], row["wthirdbar_h"]],
        [row["wfifteenbar_l"], row["wfifteenbar_r"], row["wfifteenbar_h"]],
    ]

    sim_data = simulate_correlated_triangular(
        row["comppct_r"] * 10, params, local_correlation_matrix
    )
    sim_txt = ilr_inv(sim_data[["ilr1", "ilr2"]])
    multi_sim = pd.concat([sim_data.drop(columns=["ilr1", "ilr2"]), sim_txt], axis=1)

    results.append(multi_sim)


def perturbe(x, y):
    return acomp(gsi_mul(x, y))


def gsi_mul(x, y):
    if x.shape[0] > 1 and len(x.shape) == 2 and y.shape[0] > 1 and len(y.shape) == 2:
        return x * y
    elif x.shape[0] > 1 and len(x.shape) == 2:
        return x * np.tile(y, (x.shape[0], 1))
    elif y.shape[0] > 1 and len(y.shape) == 2:
        return y * np.tile(x, (y.shape[0], 1))
    else:
        return x * y


def gsi_mul(x, y):
    # Check if x and y are 2D arrays and have more than 1 row
    x_2d_gt1 = len(x.shape) == 2 and x.shape[0] > 1
    y_2d_gt1 = len(y.shape) == 2 and y.shape[0] > 1

    # Condition 1: Both x and y are 2D arrays and have more than 1 row
    if x_2d_gt1 and y_2d_gt1:
        return x * y

    # Condition 2: x is a 2D array and has more than 1 row
    elif x_2d_gt1:
        return x * np.tile(y, (x.shape[0], 1))

    # Condition 3: y is a 2D array and has more than 1 row
    elif y_2d_gt1:
        return y * np.tile(x, (y.shape[0], 1))

    # Default: element-wise multiplication
    else:
        return x * y


# acomp function and helper functions

def gsi_plain(x):
    """
    Convert input x to its plain form.

    Parameters:
    - x: Input data, which can be a pandas DataFrame or a numpy array.

    Returns:
    - A numpy array representing the plain form of the input.
    """

    if isinstance(x, pd.DataFrame):
        return x.values
    else:
        return np.array(x)


def oneOrDataset(W, B=None):
    """
    Convert input W to a proper dataset format.

    Parameters:
    - W: Input data which can be a 1D or 2D numpy array.
    - B: An optional 2D numpy array. If provided, it influences the shape of the output.

    Returns:
    - A numpy array representing the dataset.
    """

    # Convert W to its "plain" form.
    # Note: The actual functionality of gsi.plain() is not provided,
    # so this is a placeholder and might need adjustment.
    W = gsi_plain(W)

    # If B is missing or not a 2D array
    if B is None or len(B.shape) != 2:
        if len(W.shape) == 2:
            return W
        else:
            # Convert 1D array to a 2D array with a single row
            return W.reshape(1, -1)
    else:
        if len(W.shape) == 2:
            return W
        else:
            # Convert 1D array to a 2D array with shape (len(B), len(W))
            return np.tile(W, (len(B), 1))


#############################################################
"""
'acomp' function:

"""


def acomp(X, parts=None, total=1):
    if parts is None:
        parts = list(range(X.shape[1]))

    parts = list(set(parts))

    if isinstance(X, pd.DataFrame):
        Xn = X.iloc[:, parts].to_numpy()
    else:
        Xn = X[:, parts]

    Xn /= Xn.sum(axis=1)[:, np.newaxis] / total

    return gsi_simshape(Xn, X)


def gsi_simshape(x, oldx):
    if oldx.ndim >= 2:
        return x
    return x.flatten() if oldx.ndim == 0 else x.reshape(-1)


# no longer necessary in acomp function
def gsi_plain(x):
    return x.to_numpy() if isinstance(x, pd.DataFrame) else x


def oneOrDataset(W, B=None):
    W = gsi_plain(W)

    if B is None or B.ndim != 2:
        return np.array([W]).reshape(1, -1) if W.ndim == 1 else W

    return np.tile(W, (B.shape[0], 1)) if W.ndim == 1 else W
