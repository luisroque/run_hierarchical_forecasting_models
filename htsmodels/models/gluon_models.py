import pickle
from datetime import timedelta
import os
import time
import psutil
from typing import Dict, Union, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.mx.distribution.gaussian import GaussianOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.distribution import (
    LowrankMultivariateGaussianOutput,
)
from gluonts.mx.model.deepvar import DeepVAREstimator
from gluonts.dataset.hierarchical import HierarchicalTimeSeries
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator

from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
from htsmodels.utils.logger import Logger
from htsmodels import __version__


class BaseModel(ABC):
    def __init__(self, dataset, groups, input_dir="./", n_samples=200, log_dir="."):
        self.dataset = dataset
        self.groups = groups
        self.timer_start = time.time()
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        self.input_dir = input_dir
        self.stat_cat_cardinalities = [
            v for k, v in self.groups["train"]["groups_n"].items()
        ]
        self.groups["train"]["data"] = self.groups["train"]["data"].reshape(
            -1, self.groups["train"]["s"]
        )
        self.stat_cat = np.concatenate(
            ([v.reshape(-1, 1) for k, v in self.groups["train"]["groups_idx"].items()]),
            axis=1,
        )
        self.dates = [groups["dates"][0] for _ in range(groups["train"]["s"])]
        self.n_samples = n_samples
        self.time_int = self._determine_time_interval()
        self.n_train = groups["train"]["n"]
        self.n_predict = groups["predict"]["n"]
        self.s = groups["train"]["s"]
        self.h = groups["h"]
        self._initialize_loggers(log_dir)
        self.model_version = __version__

        self.S = self.construct_aggregation_matrix()

    def _determine_time_interval(self):
        time_interval = (self.groups["dates"][1] - self.groups["dates"][0]).days
        if time_interval < 8:
            return "W"
        elif time_interval < 32:
            return "1M"
        elif time_interval < 93:
            return "3M"
        elif time_interval < 367:
            return "1Y"

    def _initialize_loggers(self, log_dir):
        self.logger = Logger(
            "train",
            algorithm=self.algorithm_name,
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )

    def _build_train_ds(self):
        """
        Constructs the training dataset.
        """
        train_target_values = self.groups["train"]["data"].T

        train_ds = ListDataset(
            [
                {
                    FieldName.TARGET: target.reshape(-1),
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: fsc,
                }
                for (target, start, fsc) in zip(
                    train_target_values, self.dates, self.stat_cat
                )
            ],
            freq=self.time_int,
        )

        return train_ds

    def _build_test_ds(self):
        """
        Constructs the testing dataset
        """
        nan_array = np.empty((self.groups["train"]["data"].shape[1], self.h))
        nan_array[:] = np.nan
        test_target_values = np.concatenate(
            (
                self.groups["train"]["data"].T,
                nan_array,
            ),
            axis=1,
        )

        test_ds = ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: fsc,
                }
                for (target, start, fsc) in zip(
                    test_target_values, self.dates, self.stat_cat
                )
            ],
            freq=self.time_int,
        )

        return test_ds

    def construct_aggregation_matrix(self):
        n_series = self.stat_cat.shape[0]
        aggregations = []

        # Add row for overall aggregation (sums across all series)
        aggregations.append([1] * n_series)

        # For each level in the hierarchy, build the aggregation rows
        for level in range(self.stat_cat.shape[1]):
            unique_cats = np.unique(self.stat_cat[:, level])
            for cat in unique_cats:
                row = [1 if sc[level] == cat else 0 for sc in self.stat_cat]
                aggregations.append(row)

        # Add the identity matrix for the base level time series (individual series)
        S_bottom = np.eye(n_series)

        S = np.vstack([aggregations, S_bottom])

        return S

    def _build_train_ds_multivariate(self):
        """
        Constructs the training dataset.
        """
        df = pd.DataFrame(
            self.groups["train"]["data"], index=self.groups["dates"][: -self.h]
        )
        df.index = pd.PeriodIndex(df.index, freq=self.time_int)

        hts_train = HierarchicalTimeSeries(
            ts_at_bottom_level=df,
            S=self.S,
        )

        return hts_train

    def _build_test_ds_multivariate(self):
        """
        Constructs the testing dataset.
        """
        df = pd.DataFrame(
            self.groups["predict"]["data_matrix"],
            index=self.groups["dates"],
        )
        df.index = pd.PeriodIndex(df.index, freq=self.time_int)

        hts_test_label = HierarchicalTimeSeries(
            ts_at_bottom_level=df,
            S=self.S,
        )

        return hts_test_label

    @staticmethod
    def _validate_param(param, valid_values):
        if param not in valid_values:
            raise ValueError(f"{param} is not a valid value")

    def store_results(
        self,
        res: np.ndarray,
        res_type: str,
        res_measure: str,
        track_mem: bool = True,
    ):
        """
        Store results

        Parameters:
            res: np array with the results with shape (n,s) - note that n depends of the res_type
            res_type: defines the type of results, could be 'fit_pred' to receive fitted values plus
                predictions or 'pred' to only store predictions
            res_measure: defines the measure to store, could be 'mean' or 'std'

        Returns:
            numpy.ndarray: Array of shape (n_samples, n_prediction_points, n_groups)
                containing the prediction samples.
        """
        """
        Store results, res_type should be used to define the type of results,
        could be 'fit_pred' to receive fitted values plus predictions or 'pred'
        to only store predictions
        """
        self._validate_param(res_type, ["fitpred", "pred"])
        self._validate_param(res_measure, ["mean", "std"])
        with open(
            f"{self.input_dir}results_{res_type}_{res_measure}_gp_cov_{self.dataset}_{self.model_version}.pickle",
            "wb",
        ) as handle:
            if track_mem:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger.info(f"Storing results used {mem:.3f} GB of RAM")
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def store_metrics(
        self,
        res: Dict[str, Dict[str, Union[float, np.ndarray]]],
        track_mem: bool = True,
    ):
        with open(
            f"{self.input_dir}metrics_gp_cov_{self.dataset}_{self.model_version}.pickle",
            "wb",
        ) as handle:
            if track_mem:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / (1024**3)
                self.logger.info(f"Storing error metrics used {mem:.3f} GB of RAM")
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(
        self,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
    ):
        calc_results = CalculateResultsBottomUp(
            predictions_mean=pred_mean,
            predictions_std=pred_std,
            groups=self.groups,
            dataset=self.dataset,
        )
        res = calc_results.calculate_metrics()
        self.wall_time_total = time.time() - self.timer_start

        res["wall_time"] = {}
        res["wall_time"]["wall_time_train"] = self.wall_time_train
        res["wall_time"]["wall_time_predict"] = self.wall_time_predict
        res["wall_time"]["wall_time_total"] = self.wall_time_total

        return res

    @abstractmethod
    def _select_distribution(self, dist: str):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class DeepAR(BaseModel):
    def __init__(self, *args, **kwargs):
        self.algorithm_name = "deepar"
        super().__init__(*args, **kwargs)

    def _select_distribution(self, dist: str):
        """
        Select the appropriate distribution based on the input string.
        """
        if dist == "NegativeBinomial":
            return NegativeBinomialOutput()
        else:
            return GaussianOutput()

    def train(self, lr=1e-3, epochs=100, dist="Gaussian"):
        """
        Train the DeepAR model with the given hyperparameters.
        """
        train_ds = self._build_train_ds()
        distribution = self._select_distribution(dist)

        estimator = DeepAREstimator(
            prediction_length=self.h,
            freq=self.time_int,
            distr_output=distribution,
            use_feat_dynamic_real=False,
            use_feat_static_cat=True,
            cardinality=self.stat_cat_cardinalities,
            trainer=Trainer(learning_rate=lr, epochs=epochs, num_batches_per_epoch=50),
        )

        model = estimator.train(train_ds)
        self.wall_time_train = time.time() - self.timer_start
        td = timedelta(seconds=int(time.time() - self.timer_start))
        self.logger.info(f"wall time train {str(td)}")
        return model

    def predict(self, model):
        timer_start = time.time()
        test_ds = self._build_test_ds()

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds, predictor=model, num_samples=self.n_samples
        )
        print("Obtaining time series predictions ...")
        forecasts = list(tqdm(forecast_it, total=len(test_ds)))

        pred_mean = np.zeros((self.n_predict, self.s))
        pred_std = np.zeros((self.n_predict, self.s))

        for ts in range(len(test_ds)):
            pred_mean[:, ts] = np.concatenate(
                (self.groups["train"]["data"][:, ts], forecasts[ts].mean), axis=0
            )
            pred_std[:, ts] = np.concatenate(
                (np.zeros((self.n_train,)), np.std(forecasts[ts].samples, axis=0)),
                axis=0,
            )

        self.wall_time_predict = time.time() - timer_start
        return pred_mean, pred_std


class DeepVARHierarchical(BaseModel):
    def __init__(self, *args, **kwargs):
        self.algorithm_name = "deepvar"
        super().__init__(*args, **kwargs)

    def _select_distribution(self, dist: str):
        pass

    def train(self, lr=1e-3, epochs=100):
        """
        Train the DeepVAR model with the given hyperparameters.
        """
        hts_train = self._build_train_ds_multivariate()
        train_ds = hts_train.to_dataset()

        estimator = DeepVARHierarchicalEstimator(
            prediction_length=self.h,
            freq=self.time_int,
            use_feat_dynamic_real=False,
            cardinality=self.stat_cat_cardinalities,
            trainer=Trainer(learning_rate=lr, epochs=epochs, num_batches_per_epoch=50),
            target_dim=hts_train.num_ts,
            S=hts_train.S,
        )

        model = estimator.train(train_ds)
        self.wall_time_train = time.time() - self.timer_start
        td = timedelta(seconds=int(time.time() - self.timer_start))
        self.logger.info(f"wall time train {str(td)}")
        return model

    def predict(self, model):
        timer_start = time.time()
        hts_test = self._build_test_ds_multivariate()
        test_ds = hts_test.to_dataset()

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds, predictor=model, num_samples=self.n_samples
        )
        print("Obtaining time series predictions ...")
        forecasts = list(tqdm(forecast_it, total=self.h))

        pred_mean = np.concatenate(
                (self.groups["train"]["data"], forecasts[0].mean[:, -self.s :]), axis=0
            )
        pred_std = np.concatenate(
                (np.zeros((self.n_train, self.s)), np.std(forecasts[0].samples, axis=0)[:, -self.s :]),
                axis=0,
            )

        self.wall_time_predict = time.time() - timer_start
        return pred_mean, pred_std
