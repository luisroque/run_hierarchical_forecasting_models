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
from sklearn.preprocessing import StandardScaler

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.mx.distribution.gaussian import GaussianOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.hierarchical import HierarchicalTimeSeries
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator

from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
from htsmodels.utils.logger import Logger
from htsmodels import __version__
import mxnet as mx
import optuna
from functools import partial
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


def get_mxnet_context():
    try:
        # Try to create a dummy array on GPU(0)
        _ = mx.nd.array([0], ctx=mx.gpu(0))
        return mx.gpu(0)
    except mx.MXNetError:
        return mx.cpu()


mxnet_context = get_mxnet_context()


class BaseModel(ABC):
    def __init__(
        self,
        dataset,
        groups,
        input_dir="./",
        n_samples=200,
        log_dir=".",
        validation_ratio=0.1,
    ):
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
        self.stat_cat_cardinalities_dict = self.groups["train"]["groups_n"]
        self.stat_cat_cardinalities_dict_idx = self.groups["train"]["groups_idx"]
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
        self.scaler = StandardScaler()

        self.S = self.construct_aggregation_matrix()
        self.seasonality = self.groups["seasonality"]

        # Split the data into training and validation sets
        train_data, validation_data, train_dates, validation_dates = self._split_data(
            groups, self.h
        )

        # store the training + validation, which in this case is the validation set
        # because GluonTS requires it to be the concatenation of both
        self.groups["train"]["dates"] = validation_dates
        # smaller subset of the data to be used for validation purposes
        self.groups["train_val"] = {"data": train_data, "dates": train_dates}
        # validation needs to be the concatenation of train and validation for GluonTS
        self.groups["validation"] = {"data": validation_data, "dates": validation_dates}
        self.mase = MeanAbsoluteScaledError(multioutput="raw_values")

    @staticmethod
    def _split_data(groups, validation_length):
        total_length = groups["train"]["data"].shape[0]
        train_length = total_length - validation_length

        train_data = groups["train"]["data"][:train_length]
        original_validation_data = groups["train"]["data"][train_length:]

        # Concatenating training data with original validation data for the new validation set
        validation_data = np.concatenate((train_data, original_validation_data))

        train_dates = groups["dates"][:train_length]
        validation_dates = groups["dates"][:total_length]

        return train_data, validation_data, train_dates, validation_dates

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

    def scale_train_data(self, data):
        """Fit and transform the training data"""
        return self.scaler.fit_transform(data)

    def scale_test_data(self, data):
        """Transform the test data using the scaler fitted on the training data"""
        return self.scaler.transform(data)

    def inverse_scale_data(self, scaled_data):
        """Inverse transform the data"""
        return self.scaler.inverse_transform(scaled_data)

    def inverse_scale_std(self, scaled_std):
        """Inverse transform the standard deviation of the predictions"""
        return scaled_std * self.scaler.scale_

    def _build_cardinality_dict(self):
        return self.groups["train"]["groups_idx"].items()

    @staticmethod
    def _wrap_transformed_features(transformed_data):
        # Determine the length of the features (number of columns in the arrays)
        feature_length = len(transformed_data[list(transformed_data.keys())[0]][0])
        wrapped_list = []

        for i in range(feature_length):
            # Create a dictionary for each feature vector
            feature_dict = {
                key: transformed_data[key][:, i] for key in transformed_data
            }
            wrapped_list.append(feature_dict)

        return wrapped_list

    def _build_train_ds(self):
        """
        Constructs the training dataset.
        """
        train_target_values = self.groups["train"]["data"].T
        train_data = []
        entry_idx = 0
        for (target, start, static_features_agg) in zip(
            train_target_values,
            self.dates,
            self.stat_cat,
        ):
            train_data.append(
                {
                    FieldName.TARGET: target.reshape(-1),
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: static_features_agg,
                    **{
                        f"{k}": np.array(v[entry_idx]).reshape(
                            -1,
                        )
                        for k, v in self.stat_cat_cardinalities_dict_idx.items()
                    },
                }
            )
            entry_idx += 1

        train_ds = ListDataset(
            train_data,
            freq=self.time_int,
        )

        return train_ds

    def _build_train_val_ds(self):
        """
        Constructs the training dataset for validation purposes.
        """
        train_target_values = self.groups["train_val"]["data"].T
        train_data = []
        entry_idx = 0
        for (target, start, static_features_agg) in zip(
            train_target_values,
            self.dates,
            self.stat_cat,
        ):
            train_data.append(
                {
                    FieldName.TARGET: target.reshape(-1),
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: static_features_agg,
                    **{
                        f"{k}": np.array(v[entry_idx]).reshape(
                            -1,
                        )
                        for k, v in self.stat_cat_cardinalities_dict_idx.items()
                    },
                }
            )
            entry_idx += 1

        train_ds = ListDataset(
            train_data,
            freq=self.time_int,
        )

        return train_ds

    def _build_validation_ds(self):
        """
        Constructs the testing dataset
        """
        nan_array = np.empty((self.groups["train_val"]["data"].shape[1], self.h))
        nan_array[:] = np.nan
        val_target_values = np.concatenate(
            (
                self.groups["train_val"]["data"].T,
                nan_array,
            ),
            axis=1,
        )

        entry_idx = 0
        val_data = []

        for (target, start, static_features_agg) in zip(
            val_target_values,
            self.dates,
            self.stat_cat,
        ):
            val_data.append(
                {
                    FieldName.TARGET: target.reshape(-1),
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: static_features_agg,
                    **{
                        f"{k}": np.array(v[entry_idx]).reshape(
                            -1,
                        )
                        for k, v in self.stat_cat_cardinalities_dict_idx.items()
                    },
                }
            )
            entry_idx += 1

        val_ds = ListDataset(
            val_data,
            freq=self.time_int,
        )

        return val_ds

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

        entry_idx = 0
        test_data = []

        for (target, start, static_features_agg) in zip(
            test_target_values,
            self.dates,
            self.stat_cat,
        ):
            test_data.append(
                {
                    FieldName.TARGET: target.reshape(-1),
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: static_features_agg,
                    **{
                        f"{k}": np.array(v[entry_idx]).reshape(
                            -1,
                        )
                        for k, v in self.stat_cat_cardinalities_dict_idx.items()
                    },
                }
            )
            entry_idx += 1

        test_ds = ListDataset(
            test_data,
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
            self.groups["train"]["data"], index=self.groups["train"]["dates"]
        )
        df.index = pd.PeriodIndex(df.index, freq=self.time_int)

        hts_train = HierarchicalTimeSeries(
            ts_at_bottom_level=df,
            S=self.S,
        )

        return hts_train

    def _build_train_val_ds_multivariate(self):
        """
        Constructs the training dataset for validation purposes.
        """
        df = pd.DataFrame(
            self.groups["train_val"]["data"], index=self.groups["train_val"]["dates"]
        )
        df.index = pd.PeriodIndex(df.index, freq=self.time_int)

        hts_train = HierarchicalTimeSeries(
            ts_at_bottom_level=df,
            S=self.S,
        )

        return hts_train

    def _build_validation_ds_multivariate(self):
        df_validation = pd.DataFrame(
            self.groups["validation"]["data"], index=self.groups["validation"]["dates"]
        )
        df_validation.index = pd.PeriodIndex(df_validation.index, freq=self.time_int)

        hts_validation = HierarchicalTimeSeries(
            ts_at_bottom_level=df_validation,
            S=self.S,
        )

        return hts_validation

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

    def optimize_hyperparams(
        self, evaluation_function, validation_data, n_trials=20, epochs=20
    ):
        def objective(trial, validation_data):
            lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
            num_layers = trial.suggest_int("num_layers", 1, 4)
            num_cells = trial.suggest_int("num_cells", 10, 40)
            cell_type = trial.suggest_categorical("cell_type", ["lstm", "gru"])
            likelihood_weight = trial.suggest_uniform("likelihood_weight", 0.0, 1.0)
            CRPS_weight = trial.suggest_uniform("CRPS_weight", 0.0, 1.0)
            embedding_dimension = trial.suggest_int("embedding_dimension", 1, 10)

            model = self.train(
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                num_layers=num_layers,
                num_cells=num_cells,
                cell_type=cell_type,
                likelihood_weight=likelihood_weight,
                CRPS_weight=CRPS_weight,
                embedding_dimension=embedding_dimension,
            )
            metric = evaluation_function(model=model, validation_data=validation_data)
            return metric

        study = optuna.create_study(direction="minimize")
        objective_with_data = partial(objective, validation_data=validation_data)
        study.optimize(objective_with_data, n_trials=n_trials)

        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best validation metric: {study.best_value}")

        return study.best_params, study.best_value


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
            trainer=Trainer(
                learning_rate=lr,
                epochs=epochs,
                num_batches_per_epoch=50,
                ctx=mxnet_context,
            ),
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

    def train(
        self,
        lr=1e-3,
        epochs=100,
        batch_size=4,
        num_layers=2,
        num_cells=40,
        cell_type="lstm",
        likelihood_weight=0.0,
        CRPS_weight=1.0,
        embedding_dimension=5,
        validation=False
    ):
        """
        Train the DeepVAR model with the given hyperparameters.
        """
        if validation:
            hts_train = self._build_train_ds_multivariate()
            train_ds = hts_train.to_dataset()
        else:
            hts_train = self._build_train_val_ds_multivariate()
            train_ds = hts_train.to_dataset()

        estimator = DeepVARHierarchicalEstimator(
            prediction_length=self.h,
            freq=self.time_int,
            use_feat_dynamic_real=False,
            cardinality=self.stat_cat_cardinalities,
            batch_size=batch_size,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            trainer=Trainer(
                learning_rate=lr,
                epochs=epochs,
                num_batches_per_epoch=50,
                ctx=mxnet_context,
            ),
            target_dim=hts_train.num_ts,
            S=hts_train.S,
            likelihood_weight=likelihood_weight,
            CRPS_weight=CRPS_weight,
            embedding_dimension=embedding_dimension,
        )

        model = estimator.train(train_ds)
        self.wall_time_train = time.time() - self.timer_start
        td = timedelta(seconds=int(time.time() - self.timer_start))
        self.logger.info(f"wall time train {str(td)}")
        return model

    def evaluate(self, model, validation_data):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=validation_data, predictor=model, num_samples=self.n_samples
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        forecast_means = np.array([f.mean.squeeze() for f in forecasts]).squeeze()
        true_targets = np.array([ts.values for ts in tss]).squeeze()[-self.h:]

        forecast_df = pd.DataFrame(forecast_means)
        target_df = pd.DataFrame(true_targets)

        y_train = np.dot(self.groups["train"]["data"], self.S.T)

        if self.groups["train"]["data"].shape[0] < self.seasonality:
            sp = 1  # non-seasonal case, use a lag of 1
        else:
            sp = self.seasonality
        mase = MeanAbsoluteScaledError(multioutput="raw_values")
        mase_score = mase(
            y_true=target_df,
            y_pred=forecast_df,
            y_train=y_train,
            sp=sp,
        )

        return mase_score.mean()

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
            (
                np.zeros((self.n_train, self.s)),
                np.std(forecasts[0].samples, axis=0)[:, -self.s :],
            ),
            axis=0,
        )

        self.wall_time_predict = time.time() - timer_start
        return pred_mean, pred_std

    def hyper_tuning(self, n_trials=20, epochs=50):
        hts_val = self._build_validation_ds_multivariate()
        val_ds = hts_val.to_dataset()

        hyperparams_optimized, validation_loss = self.optimize_hyperparams(
            evaluation_function=self.evaluate,
            validation_data=val_ds,
            n_trials=n_trials,
            epochs=epochs,
        )


class TFT(BaseModel):
    def __init__(self, *args, **kwargs):
        self.algorithm_name = "tft"
        super().__init__(*args, **kwargs)

    def _select_distribution(self, dist: str):
        pass

    def train(self, lr=1e-3, epochs=100, dist="Gaussian"):
        """
        Train the TFT model with the given hyperparameters.
        """
        train_ds = self._build_train_ds()

        estimator = TemporalFusionTransformerEstimator(
            prediction_length=self.h,
            freq=self.time_int,
            static_cardinalities=self.stat_cat_cardinalities_dict,
            trainer=Trainer(
                learning_rate=lr,
                epochs=epochs,
                num_batches_per_epoch=50,
                ctx=mxnet_context,
            ),
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

        for ts, forecast in enumerate(forecasts):
            # Concatenate the mean predictions
            if hasattr(forecast, "mean"):
                pred_mean[:, ts] = np.concatenate(
                    (self.groups["train"]["data"][:, ts], forecast.mean), axis=0
                )

            # Estimate the standard deviation if forecast is a QuantileForecast
            if hasattr(forecast, "quantile"):
                q10 = forecast.quantile("0.1")
                q90 = forecast.quantile("0.9")
                spread = q90 - q10
                estimated_std_dev = spread / 1.645
                pred_std[:, ts] = np.concatenate(
                    (np.zeros((self.n_train,)), estimated_std_dev), axis=0
                )
            else:
                # Fall back to standard deviation of samples if available
                if hasattr(forecast, "samples"):
                    pred_std[:, ts] = np.concatenate(
                        (np.zeros((self.n_train,)), np.std(forecast.samples, axis=0)),
                        axis=0,
                    )

        self.wall_time_predict = time.time() - timer_start
        return pred_mean, pred_std
