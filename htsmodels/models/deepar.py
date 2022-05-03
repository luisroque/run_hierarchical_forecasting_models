from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm import tqdm
from htsmodels.results.calculate_metrics import CalculateResultsBottomUp
import numpy as np
import pickle
from pathlib import Path
import time


class DeepAR:

    def __init__(self, dataset, groups, input_dir='./', n_samples=500,
                 store_prediction_samples=False,
                 store_prediction_points=False):
        self.dataset = dataset
        self.groups = groups
        self.n_samples = n_samples
        self.timer_start = time.time()
        self.wall_time_preprocess = None
        self.wall_time_build_model = None
        self.wall_time_train = None
        self.wall_time_predict = None
        self.wall_time_total = None
        self.input_dir = input_dir
        self._create_directories()
        self.stat_cat_cardinalities = [v for k, v in self.groups['train']['groups_n'].items()]
        self.stat_cat = np.concatenate(([v.reshape(-1, 1) for k, v in self.groups['train']['groups_idx'].items()]),
                                       axis=1)
        self.dates = [groups['dates'][0] for _ in range(groups['train']['s'])]
        self.store_prediction_samples = store_prediction_samples
        self.store_prediction_points = store_prediction_points

        time_interval = (self.groups['dates'][1] - self.groups['dates'][0]).days
        if time_interval < 8:
            self.time_int = 'W'
        elif time_interval < 32:
            self.time_int = 'M'
        elif time_interval < 93:
            self.time_int = 'Q'
        elif time_interval < 367:
            self.time_int = 'Y'

    def _create_directories(self):
        # Create directory to store results if does not exist
        Path(f'{self.input_dir}results').mkdir(parents=True, exist_ok=True)

    def _build_train_ds(self):
        train_target_values = self.groups['train']['data'].T

        train_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for (target, start, fsc) in zip(train_target_values,
                                            self.dates,
                                            self.stat_cat)
        ], freq=self.time_int)

        return train_ds

    def _build_test_ds(self):
        test_target_values = self.groups['predict']['data'].reshape(self.groups['predict']['s'],
                                                                    self.groups['predict']['n'])

        test_ds = ListDataset([
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for (target, start, fsc) in zip(test_target_values,
                                            self.dates,
                                            self.stat_cat)
        ], freq=self.time_int)

        return test_ds

    def train(self, lr=1e-3, epochs=100):
        self.wall_time_preprocess = time.time() - self.timer_start
        train_ds = self._build_train_ds()
        self.wall_time_build_model = time.time() - self.timer_start - self.wall_time_preprocess

        estimator = DeepAREstimator(
            prediction_length=self.groups['h'],
            freq=self.time_int,
            distr_output=NegativeBinomialOutput(),
            use_feat_dynamic_real=False,
            use_feat_static_cat=True,
            cardinality=self.stat_cat_cardinalities,
            trainer=Trainer(
                learning_rate=lr,
                epochs=epochs,
                num_batches_per_epoch=50,
                batch_size=32
            )
        )

        model = estimator.train(train_ds)
        self.wall_time_train = time.time() - self.timer_start - self.wall_time_build_model
        return model

    def predict(self, model):
        test_ds = self._build_test_ds()

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=model,
            num_samples=self.n_samples
        )

        print("Obtaining time series predictions ...")
        forecasts = list(tqdm(forecast_it, total=len(test_ds)))
        self.wall_time_predict = time.time() - self.timer_start - self.wall_time_train

        return forecasts

    def results(self, forecasts):
        res = np.zeros((len(forecasts), self.n_samples, self.groups['h']))
        for i, j in enumerate(forecasts):
            res[i] = j.samples

        res = np.transpose(np.concatenate((np.zeros((self.groups['train']['s'], self.n_samples, self.groups['train']['n']),
                                                    dtype=np.float64), res),
                                          axis=2),
                           (2, 0, 1))

        return res

    def store_metrics(self, res):
        with open(f'{self.input_dir}results/results_gp_cov_{self.dataset}.pickle', 'wb') as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def metrics(self, samples):
        calc_results = CalculateResultsBottomUp(samples, self.groups,
                                                self.store_prediction_samples,
                                                self.store_prediction_points)
        res = calc_results.calculate_metrics()
        self.wall_time_total = time.time() - self.timer_start

        res['wall_time'] = {}
        res['wall_time']['wall_time_preprocess'] = self.wall_time_preprocess
        res['wall_time']['wall_time_build_model'] = self.wall_time_build_model
        res['wall_time']['wall_time_train'] = self.wall_time_train
        res['wall_time']['wall_time_predict'] = self.wall_time_predict
        res['wall_time']['wall_time_total'] = self.wall_time_total

        return res
