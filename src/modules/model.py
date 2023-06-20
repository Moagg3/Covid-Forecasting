import numpy as np
import pandas as pd
import torch
from statsmodels.regression.linear_model import OLS
from torch import nn


class LamposModel:
    def __init__(self) -> None:
        """Implementation of the Lampos model for denoising symptom trends with media data.
           Reference: Vasileios Lampos, Maimuna S. Majumder, Elad Yom-Tov, Michael Edelstein,
                      Simon Moura, Yohhei Hamada, Molebogeng X. Rangaka, Rachel A. McKendry,
                      and Ingemar J. Cox. 2021. Tracking COVID-19 using online search.
                      npj Digit. Med. 4, 1 (February 2021), 1–11.
                      DOI:https://doi.org/10.1038/s41746-021-00384-w

        Attributes:
            model_1 (statsmodels.regression.linear_model.OLS): AR(g) model
            model_2 (statsmodels.regression.linear_model.OLS): AR(g, m) model
            model_1_fit_results (statsmodels.regression.linear_model.RegressionResultsWrapper): AR(g) model fit results
            model_2_fit_results (statsmodels.regression.linear_model.RegressionResultsWrapper): AR(g, m) model fit results

        Example Usage:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from model import LamposModel

            >>> symptom_data = pd.read_csv("./datasets/combined.csv")
            >>> media_data = pd.read_csv("./datasets/media_count_ratio_all_2021.csv")
            >>> symptom = "symptom:shortness of breath"

            >>> g_prev_1_data = symptom_data[symptom][:-2].reset_index(drop=True)
            >>> g_prev_2_data = symptom_data[symptom][1:-1].reset_index(drop=True)
            >>> m_prev_1_data = media_data[symptom][:-2].reset_index(drop=True)
            >>> m_prev_2_data = media_data[symptom][1:-1].reset_index(drop=True)

            >>> X = pd.concat([g_prev_1_data, g_prev_2_data, m_prev_1_data, m_prev_2_data], axis=1)
            >>> X.columns = ["g_prev_1", "g_prev_2", "m_prev_1", "m_prev_2"]
            >>> y = symptom_data[symptom][2:].reset_index(drop=True).to_frame()
            >>> y.columns = ["y"]

            >>> model = LamposModel()
            >>> model.fit(X, y, g_prev_1="g_prev_1", g_prev_2="g_prev_2", m_prev_1="m_prev_1", m_prev_2="m_prev_2")
            >>> denoised_y = model.predict(X, y)
        """
        self.model_1 = None
        self.model_2 = None
        self.model_1_fit_results = None
        self.model_2_fit_results = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        g_prev_1: str = None,
        g_prev_2: str = None,
        m_prev_1: str = None,
        m_prev_2: str = None,
    ) -> None:
        """Fits the Lampos model to the data.

        Args:
            X (pandas.DataFrame): A 4-column reindexed dataframe including g_{t - 1}, g_{t - 2}, m_{t - 1}, and m_{t - 2}.
            y (pandas.DataFrame): A 1-column reindexed dataframe including g_{t}.
            g_prev_1 (str): The column header for g_{t - 1} in X.
            g_prev_2 (str): The column header for g_{t - 2} in X.
            m_prev_1 (str): The column header for m_{t - 1} in X.
            m_prev_2 (str): The column header for m_{t - 2} in X.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of rows. Found X of length {} and y of length {}".format(
            len(X), len(y)
        )
        assert y.shape[1] == 1, "y must be a column vector. Found y of shape {}".format(
            y.shape
        )
        if not self.model_1:
            # Sanity check on headers
            self.headers = [g_prev_1, g_prev_2, m_prev_1, m_prev_2]
            assert set(self.headers).issubset(
                X.columns
            ), "X must contain all headers in {}".format(self.headers)
            y = y.to_numpy().reshape(-1, 1)
            # Fit AR(g) model
            X_1 = X[self.headers[:2]].to_numpy()
            self.model_1 = OLS(y, X_1)
            self.model_1_fit_results = self.model_1.fit()
            # Fit AR(g, m) model
            X_2 = X[self.headers].to_numpy()
            self.model_2 = OLS(y, X_2)
            self.model_2_fit_results = self.model_2.fit()
        else:
            print("[WARNING] Model already fitted. Skipping fit.")

    def predict(self, X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        """Given g_{t - 1}, g_{t - 2}, m_{t - 1}, and m_{t - 2}, return a denoised g_{t}.

        Args:
            X (pandas.DataFrame): A 4-column reindexed dataframe including g_{t - 1}, g_{t - 2}, m_{t - 1}, and m_{t - 2}.
            y (pandas.DataFrame): A 1-column reindexed dataframe including g_{t}.

        Returns:
            np.ndarray: A denoised g_{t}.
        """
        assert (
            self.model_1 and self.model_2
        ), "Model not fitted. Please fit model with fit() before predicting."
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of rows. Found X of length {} and y of length {}".format(
            len(X), len(y)
        )
        assert y.shape[1] == 1, "y must be a column vector. Found y of shape {}".format(
            y.shape
        )
        assert set(self.headers).issubset(
            X.columns
        ), "X must contain all headers in {}".format(self.headers)
        # Prepare X_1, X_2, and y
        X_1 = X[self.headers[:2]].to_numpy()
        X_2 = X[self.headers].to_numpy()
        y = y.to_numpy().flatten()
        # Generate predictions
        y_pred_1 = np.concatenate([self.model_1_fit_results.predict(x) for x in X_1])
        y_pred_2 = np.concatenate([self.model_2_fit_results.predict(x) for x in X_2])
        # Calculate absolute errors
        epison_1 = np.abs(y - y_pred_1)
        epison_2 = np.abs(y - y_pred_2)
        # Denoise the symptom data
        gamma = np.where(epison_2 < epison_1, epison_2 / epison_1, 1)
        denoised_y = gamma * y
        return denoised_y


class LamposExtendedModel(LamposModel):
    """Extended Lampos model that accepts an aribitrary lag of k days.

    Args:
        LamposModel (class): The base Lampos model.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> None:
        """Fits the extended Lampos model to the data.

        Args:
            X (pandas.DataFrame): A (2 * k)-column reindexed dataframe including
                                  g_{t - 1}, g_{t - 2}, ... g{t - k}, m_{t - 1}, m_{t - 2}, ..., and m_{t - k}.
            y (pandas.DataFrame): A 1-column reindexed dataframe including g_{t}.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of rows. Found X of length {} and y of length {}".format(
            len(X), len(y)
        )
        assert y.shape[1] == 1, "y must be a column vector. Found y of shape {}".format(
            y.shape
        )
        if not self.model_1:
            # Sanity check on headers
            assert (
                X.shape[1] % 2 == 0
            ), "X must have an even number of columns. Found {}.".format(X.shape[1])
            self.k = X.shape[1] // 2
            self.headers = X.columns
            y = y.to_numpy().reshape(-1, 1)
            # Fit AR(g) model
            X_1 = X[self.headers[: self.k]].to_numpy()
            self.model_1 = OLS(y, X_1)
            self.model_1_fit_results = self.model_1.fit()
            # Fit AR(g, m) model
            X_2 = X[self.headers].to_numpy()
            self.model_2 = OLS(y, X_2)
            self.model_2_fit_results = self.model_2.fit()
        else:
            print("[WARNING] Model already fitted. Skipping fit.")

    def predict(self, X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        """Given g_{t - 1}, g_{t - 2}, ... g{t - k}, m_{t - 1}, m_{t - 2}, ..., and m_{t - k},
           return a denoised g_{t}.

        Args:
            X (pandas.DataFrame): A 4-column reindexed dataframe including g_{t - 1}, g_{t - 2}, m_{t - 1}, and m_{t - 2}.
            y (pandas.DataFrame): A 1-column reindexed dataframe including g_{t}.

        Returns:
            np.ndarray: A denoised g_{t}.
        """
        assert (
            self.model_1 and self.model_2
        ), "Model not fitted. Please fit model with fit() before predicting."
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of rows. Found X of length {} and y of length {}".format(
            len(X), len(y)
        )
        assert y.shape[1] == 1, "y must be a column vector. Found y of shape {}".format(
            y.shape
        )
        assert set(self.headers).issubset(
            X.columns
        ), "X must contain all headers in {}".format(self.headers)
        # Prepare X_1, X_2, and y
        X_1 = X[self.headers[: self.k]].to_numpy()
        X_2 = X[self.headers].to_numpy()
        y = y.to_numpy().flatten()
        # Generate predictions
        y_pred_1 = np.concatenate([self.model_1_fit_results.predict(x) for x in X_1])
        y_pred_2 = np.concatenate([self.model_2_fit_results.predict(x) for x in X_2])
        # Calculate absolute errors
        epison_1 = np.abs(y - y_pred_1)
        epison_2 = np.abs(y - y_pred_2)
        # Denoise the symptom data
        gamma = np.where(epison_2 < epison_1, epison_2 / epison_1, 1)
        denoised_y = gamma * y
        return denoised_y


class ShallowRegressionLSTM(nn.Module):
    """A shallow regression LSTM model.
    Reference: https://www.crosstab.io/articles/time-series-pytorch-lstm/
    """

    def __init__(self, num_features: int, hidden_units: int) -> None:
        """Initializes the model.

        Args:
            num_features (int): The number of features in the input.
                                The default should be 2: search trend time series and media time series.
            hidden_units (int): The number of hidden units in the LSTM layer.
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): A 3D tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: A 2D tensor of shape (batch_size, 1).
        """
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of H_n is num_layers, which is set to 1 above.

        return out
