import pickle
from datetime import timedelta
import os
import time
from pathlib import Path
import psutil
from typing import Dict, Union, Tuple

import torch
import numpy as np
import gpytorch
from sklearn.preprocessing import StandardScaler
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import LikelihoodList
from gpytorch.mlls import SumMarginalLogLikelihood

from htsmodels.utils.logger import Logger
from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
from htsmodels.models.gp import ExactGPModel
from htsmodels.models.mean_functions import PiecewiseLinearMean, LinearMean, ZeroMean
from htsmodels import __version__


class SGP:
    def __init__(self, dataset, groups, input_dir="./", n_samples=500, log_dir="."):
        self.dataset = dataset
        self.groups = groups
        self.input_dir = input_dir
        self.timer_start = time.time()
        self.wall_time_preprocess = None
        self.wall_time_build_model = None
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        self.n_samples = n_samples

        self.groups, self.scaler = self._preprocess(groups)

        self.train_x = torch.arange(groups["train"]["n"])
        self.train_x = self.train_x.type(torch.DoubleTensor)
        self.train_x = self.train_x.unsqueeze(-1)
        self.train_y = torch.from_numpy(groups["train"]["data"])

        self.original_data_transformed = groups["predict"]["data_matrix"]

        self.n_train = groups["train"]["n"]
        self.n_predict = groups["predict"]["n"]
        self.s = groups["train"]["s"]

        self.logger_train = Logger(
            "train",
            algorithm="standard_gp",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )
        self.logger_predict = Logger(
            "predict",
            algorithm="standard_gp",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )
        self.logger_metrics = Logger(
            "metrics",
            algorithm="standard_gp",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )

        self.model_version = __version__

    def _preprocess(self, groups):
        scaler = StandardScaler()
        scaler.fit(self.groups["train"]["data"])
        groups["train"]["data"] = scaler.transform(groups["train"]["data"])
        groups["predict"]["data_matrix"] = scaler.transform(
            groups["predict"]["data_matrix"]
        )

        return groups, scaler

    def _build_cov_matrices(self):
        covs = []
        for i in range(self.groups["train"]["s"]):
            # RBF kernel
            rbf_kernel = gpytorch.kernels.RBFKernel()
            rbf_kernel.lengthscale = torch.tensor([1.0])
            scale_rbf_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
            scale_rbf_kernel.outputscale = torch.tensor([0.5])

            # Periodic Kernel
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            periodic_kernel.period_length = torch.tensor([self.groups["seasonality"]])
            periodic_kernel.lengthscale = torch.tensor([0.5])
            scale_periodic_kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)
            scale_periodic_kernel.outputscale = torch.tensor([1.5])

            # Cov Matrix
            cov = scale_rbf_kernel + scale_periodic_kernel
            covs.append(cov)

        return covs

    def _build_model(self, mean_function):
        covs = self._build_cov_matrices()
        n_changepoints = 4
        changepoints = np.linspace(0, self.groups["train"]["n"], n_changepoints + 2)[
            1:-1
        ]

        model_list = []
        likelihood_list = []
        for i in range(self.groups["train"]["s"]):
            likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
            if mean_function == "piecewise_linear":
                model_list.append(
                    ExactGPModel(
                        self.train_x,
                        self.train_y[:, i],
                        likelihood_list[i],
                        covs[i],
                        PiecewiseLinearMean(changepoints),
                    )
                )
            elif mean_function == "zero":
                model_list.append(
                    ExactGPModel(
                        self.train_x,
                        self.train_y[:, i],
                        likelihood_list[i],
                        covs[i],
                        gpytorch.means.ZeroMean(),
                    )
                )
            elif mean_function == "linear":
                model_list.append(
                    ExactGPModel(
                        self.train_x,
                        self.train_y[:, i],
                        likelihood_list[i],
                        covs[i],
                        LinearMean(1),
                    )
                )

        return likelihood_list, model_list

    def train(
        self,
        n_iterations=150,
        lr=1e-3,
        mean_function="piecewise_linear",
        track_mem=True,
    ):
        likelihood_list, model_list = self._build_model(mean_function)

        model = gpytorch.models.IndependentModelList(*model_list)
        likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        mll = SumMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr
        )  # Includes GaussianLikelihood parameters

        for i in range(n_iterations):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, n_iterations, loss.item()))
            optimizer.step()

            if i % 30 == 0 and track_mem:
                # Track RAM usage
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger_train.info(f"train used {mem:.3f} GB of RAM")

        self.wall_time_train = time.time() - self.timer_start
        td = timedelta(seconds=int(time.time() - self.timer_start))
        self.logger_train.info(f"Num epochs {i}")
        self.logger_train.info(f"wall time train {str(td)}")

        return model, likelihood

    def predict(
        self,
        model: IndependentModelList,
        likelihood: LikelihoodList,
        track_mem: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the model.

        Parameters:
            model: The GP model.
            likelihood: The likelihood function.
            track_mem: Track and log RAM usage

        Returns:
            numpy.ndarray: Array of shape (n_samples, n_prediction_points, n_groups)
                containing the prediction samples.
        """
        timer_start = time.time()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.arange(self.groups["predict"]["n"]).type(torch.DoubleTensor)
            predictions = likelihood(
                *model(*[test_x for i in range(self.groups["predict"]["s"])])
            )
            if track_mem:
                # Track RAM usage
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger_predict.info(f"predict used {mem:.3f} GB of RAM")

        pred_mean_scaled = np.zeros((self.n_predict, self.s))
        pred_std_scaled = np.zeros((self.n_predict, self.s))
        for ts in range(self.s):
            pred_mean_scaled[:, ts] = predictions[ts].mean.detach().numpy()
            pred_std_scaled[:, ts] = np.sqrt(predictions[ts].variance.detach().numpy())

        # transform back the data
        pred_mean = pred_mean_scaled * self.scaler.scale_ + self.scaler.mean_
        pred_std = pred_std_scaled * self.scaler.scale_

        self.wall_time_predict = time.time() - timer_start
        return (pred_mean, pred_std), (pred_mean_scaled, pred_std_scaled)

    def store_metrics(
        self,
        res: Dict[str, Dict[str, Union[float, np.ndarray]]],
        track_mem: bool = True,
    ):
        with open(
            f"{self.input_dir}results_gp_cov_{self.dataset}_{self.model_version}.pickle",
            "wb",
        ) as handle:
            if track_mem:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger_metrics.info(
                    f"Storing error metrics used {mem:.3f} GB of RAM"
                )
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(
        self,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        track_mem: bool = True,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Calculate evaluation metrics for the predictions.

        Parameters:
            pred_mean: Array of shape (n_prediction_points, n_series)
                containing the prediction samples.
            pred_std: Array of shape (n_prediction_points, n_series)
                containing the prediction samples.
            track_mem: Track and log RAM usage

        Returns:
            dict: Dictionary with the evaluation metrics. The keys are the metric names,
                and the values are dictionaries with the results for each group.
        """
        calc_results = CalculateResultsBottomUp(
            predictions_mean=pred_mean,
            predictions_std=pred_std,
            groups=self.groups,
            dataset=self.dataset,
        )
        res = calc_results.calculate_metrics()
        if track_mem:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024**3)
            self.logger_metrics.info(
                f"calculating error metrics used {mem:.3f} GB of RAM"
            )
        self.wall_time_total = time.time() - self.timer_start

        res["wall_time"] = {}
        res["wall_time"]["wall_time_train"] = self.wall_time_train
        res["wall_time"]["wall_time_predict"] = self.wall_time_predict
        res["wall_time"]["wall_time_total"] = self.wall_time_total

        return res
