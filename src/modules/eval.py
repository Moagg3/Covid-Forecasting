from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import sklearn
import git
from fastdtw import fastdtw
from numpy.lib import pad
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

REPO_ROOT = git.Repo(".", search_parent_directories=True).working_tree_dir


def dynamic_time_warping(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
    distance: scipy.spatial.distance = euclidean,
    scaler: sklearn.preprocessing = MinMaxScaler,
) -> float:
    """Calulate the dynamic time warping distance between two time series.

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.
        distance (function): A distance function.
        scaler (sklearn.preprocessing): A scaler function.

    Returns:
        float: The dynamic time warping distance.
    """
    if isinstance(x1, pd.Series):
        x1 = x1.to_numpy()
    elif isinstance(x1, list):
        x1 = np.array(x1)

    if isinstance(x2, pd.Series):
        x2 = x2.to_numpy()
    elif isinstance(x2, list):
        x2 = np.array(x2)

    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)

    if scaler:
        x1_scaler, x2_scaler = scaler(), scaler()
        x1 = x1_scaler.fit_transform(x1)
        x2 = x2_scaler.fit_transform(x2)

    dist, _ = fastdtw(x1, x2, dist=distance)

    return dist


def calculate_pearsonr(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
    confidence: float = 0.95,
    return_confidence_interval: bool = False,
) -> List[float]:
    """Calulate the Pearson correlation between two time series.

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.
        confidence (float): The confidence level for the confidence interval.
        return_confidence_interval (bool): Whether to return the confidence interval.

    Returns:
        List[float]: The Pearson correlation, p-value, and, optionally, confidence interval.
    """
    assert x1 is not None and len(x1) > 0, "x1 must be a non-empty array."
    assert x2 is not None and len(x2) > 0, "x2 must be a non-empty array."
    assert len(x1) == len(x2), "x1 and x2 must be the same length."

    result = scipy.stats.pearsonr(x1, x2)
    statistic = result[0]
    pval = result[1]

    if not return_confidence_interval:
        return statistic, pval

    confidence_interval = result.confidence_interval(confidence)
    lower_bound = confidence_interval.low
    upper_bound = confidence_interval.high

    return statistic, pval, lower_bound, upper_bound


def calculate_spearmanr(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
) -> Tuple[Union[float, np.ndarray], float]:
    """Calulate the Spearman correlation between two time series.

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.

    Returns:
        Tuple[Union[float, np.ndarray], float]: The Spearman correlation and p-value.
    """
    assert x1 is not None and len(x1) > 0, "x1 must be a non-empty array."
    assert x2 is not None and len(x2) > 0, "x2 must be a non-empty array."
    assert len(x1) == len(x2), "x1 and x2 must be the same length."

    rho, pval = spearmanr(x1, x2)

    return rho, pval


def calculate_rolling_window_pearsonr(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
    window: int = 5,
) -> np.ndarray:
    """Calculate the rolling window Pearson correlation between two time series.

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.
        window (int): The window size.

    Returns:
        np.ndarray: The rolling window Pearson correlation.
    """
    assert x1 is not None and len(x1) > 0, "x1 must be a non-empty array."
    assert x2 is not None and len(x2) > 0, "x2 must be a non-empty array."
    assert len(x1) == len(x2), "x1 and x2 must be the same length."
    assert window > 0, "window must be greater than 0."

    if not isinstance(x1, pd.Series):
        x1 = pd.Series(x1)
    if not isinstance(x2, pd.Series):
        x2 = pd.Series(x2)

    corrs = x1.rolling(window).corr(x2).to_numpy()

    return corrs


def calculate_rolling_window_spearmanr(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
    window: int = 5,
) -> np.ndarray:
    """Calulate the rolling window Spearman correlation between two time series.
       Reference: https://stackoverflow.com/a/48211159/11526586

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.
        window (int): The window size.

    Returns:
        np.ndarray: The rolling window Spearman correlation.
    """
    assert x1 is not None and len(x1) > 0, "x1 must be a non-empty array."
    assert x2 is not None and len(x2) > 0, "x2 must be a non-empty array."
    assert len(x1) == len(x2), "x1 and x2 must be the same length."
    assert window > 0, "window must be greater than 0."

    if not isinstance(x1, np.ndarray):
        x1 = pd.Series(x1).to_numpy()

    if not isinstance(x2, np.ndarray):
        x2 = pd.Series(x2).to_numpy()

    stride_x1 = x1.strides[0]
    ssa = as_strided(
        x1, shape=[len(x1) - window + 1, window], strides=[stride_x1, stride_x1]
    )
    stride_x2 = x2.strides[0]
    ssb = as_strided(
        x2, shape=[len(x2) - window + 1, window], strides=[stride_x2, stride_x2]
    )
    r_x1 = pd.DataFrame(ssa)
    r_x2 = pd.DataFrame(ssb)
    r_x1 = r_x1.rank(1)
    r_x2 = r_x2.rank(1)

    corrs = r_x1.corrwith(r_x2, axis=1, method="spearman")
    corrs = pad(corrs, (window - 1, 0), "constant", constant_values=np.nan)

    return corrs


def apply_moving_average(
    x: Union[list, np.ndarray, pd.Series],
    window: int = 5,
) -> pd.Series:
    """Apply a moving average to the dataframe.

    Args:
        x (list or np.ndarray or pd.Series): A 1D array of time series data.
        window (int): The window size.

    Returns:
        pd.Series: The moving average of x.
    """
    assert x is not None and len(x) > 0, "x must be a non-empty array."
    assert window > 0, "window must be greater than 0."

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    moving_avg_x = x.rolling(window=window).mean()

    return moving_avg_x


def calculate_rmse(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
) -> List[float]:
    """Calulate RMSE between x1 (actual) and x2 (predictions)

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.

    Returns:
        float: RMSE between x1 and x2
    """
    assert x1 is not None and len(x1) > 0, "x1 must be a non-empty array."
    assert x2 is not None and len(x2) > 0, "x2 must be a non-empty array."
    assert len(x1) == len(x2), "x1 and x2 must be the same length."

    rmse = mean_squared_error(x1, x2, squared=False)

    return rmse


def run_eval_suite_default(
    x1: Union[list, np.ndarray, pd.Series],
    x2: Union[list, np.ndarray, pd.Series],
) -> List[float]:
    """Runs evaluation suite for rmse, dtw, spearman, and pearson with default params.

    Args:
        x1 (list, np.ndarray or pd.Series): A 1D array of time series data.
        x2 (list, np.ndarray or pd.Series): A 1D array of time series data.

    Returns:
        dict[string, float]: Dictionary with scores for each metric
    """
    rmse = calculate_rmse(x1, x2)
    dtw_dist = dynamic_time_warping(x1, x2)
    pearsonr = calculate_pearsonr(x1, x2)
    pearson_statistic = pearsonr[0]
    pearson_pval = pearsonr[1]
    spearmanr = calculate_spearmanr(x1, x2)
    spearman_rho = spearmanr[0]
    spearman_pval = spearmanr[1]
    eval_dict = {
        "rmse": rmse,
        "dtw distance": dtw_dist,
        "pearson statistic": pearson_statistic,
        "pearson p-value": pearson_pval,
        "spearman rho": spearman_rho,
        "spearman p-value": spearman_pval,
    }
    return eval_dict


def get_model_results(predictions, actuals, model, media=False, lead=0, save=True):
    """Saves model results with date, predicted_case_count, and actual_case_count columns.
       Saves to results/ if save=True.

    Args:
        predictions (list, np.ndarray or pd.Series): predicted case counts
        actuals (list, np.ndarray or pd.Series): actual case counts
        model (string): model name
        media (boolean): whether or not media data is passed alongside rsv data as input to model
        lead (int): lead time of predictions if applicable
        save (boolean): whether or not to save results

    Returns:
        pd.DataFrame: df containing results that were saved
    """

    assert len(predictions) == len(
        actuals
    ), "predictions and actuals must be of same length"

    dates = pd.read_csv(f"{REPO_ROOT}/src/datasets/combined.csv")["date"].to_numpy()
    results_df = pd.DataFrame(
        {
            "date": dates[len(dates) - len(predictions) :],
            "predicted_case_count": predictions,
            "actual_case_count": actuals,
        }
    )
    media = "_media" if media else ""
    lead = "" if lead == 0 else f"_lead{str(lead)}"
    if save:
        save_path = f"{REPO_ROOT}/src/results/{model}{media}{lead}.csv"
        results_df.to_csv(save_path, index=False)
    return results_df
