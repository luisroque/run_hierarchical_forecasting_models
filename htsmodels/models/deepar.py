import pickle
from datetime import timedelta
import os
import time
from pathlib import Path
import psutil

import numpy as np
from tqdm import tqdm

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.mx.distribution.gaussian import GaussianOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
from htsmodels.utils.logger import Logger


class DeepAR:
    def __init__(
        self,
        dataset,
        groups,
        input_dir="./",
        n_samples=200,
        log_dir: str = ".",
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
        self.stat_cat = np.concatenate(
            ([v.reshape(-1, 1) for k, v in self.groups["train"]["groups_idx"].items()]),
            axis=1,
        )
        self.dates = [groups["dates"][0] for _ in range(groups["train"]["s"])]
        self.n_samples = n_samples

        time_interval = (self.groups["dates"][1] - self.groups["dates"][0]).days
        if time_interval < 8:
            self.time_int = "W"
        elif time_interval < 32:
            self.time_int = "M"
        elif time_interval < 93:
            self.time_int = "Q"
        elif time_interval < 367:
            self.time_int = "Y"

        self.n_train = groups["train"]["n"]
        self.n_predict = groups["predict"]["n"]
        self.s = groups["train"]["s"]
        self.h = groups["h"]

        self.logger_train = Logger(
            "train",
            algorithm="deepar",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )
        self.logger_predict = Logger(
            "predict",
            algorithm="deepar",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )
        self.logger_metrics = Logger(
            "metrics",
            algorithm="deepar",
            dataset=self.dataset,
            to_file=True,
            log_dir=log_dir,
        )

    def _build_train_ds(self):
        train_target_values = self.groups["train"]["data"].T

        train_ds = ListDataset(
            [
                {
                    FieldName.TARGET: target,
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
        test_target_values = self.groups["predict"]["data"].reshape(
            self.groups["predict"]["s"], self.groups["predict"]["n"]
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

    def train(self, lr=1e-3, epochs=100, dist="Gaussian"):
        train_ds = self._build_train_ds()

        if dist == "NegativeBinomial":
            distribution = NegativeBinomialOutput()
        else:
            distribution = GaussianOutput()

        estimator = DeepAREstimator(
            prediction_length=self.groups["h"],
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
        self.logger_train.info(f"wall time train {str(td)}")
        return model

    def predict(self, model, track_mem=True):
        timer_start = time.time()
        test_ds = self._build_test_ds()

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds, predictor=model, num_samples=self.n_samples
        )

        if track_mem:
            # Track RAM usage
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024**3)
            self.logger_predict.info(f"predict used {mem:.3f} GB of RAM")

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

    def store_metrics(self, res, track_mem=True):
        with open(
            f"{self.input_dir}results_gp_cov_{self.dataset}.pickle", "wb"
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
    ):
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
