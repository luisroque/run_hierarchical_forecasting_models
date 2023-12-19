import unittest
import tsaugmentation as tsag
from htsmodels.models.gluon_models import DeepVARHierarchical
from htsmodels import __version__

import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from sklearn.metrics import mean_absolute_error


class TestModel(unittest.TestCase):

    def setUp(self):
        self.dataset_name = 'prison'
        self.data = tsag.preprocessing.PreprocessDatasets(self.dataset_name, 'Q').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['predict']['s']
        self.h = self.data['h']
        self.version = __version__
        self.deepvar = DeepVARHierarchical(dataset=self.dataset_name, groups=self.data)

    def test_correct_train(self):
        model = self.deepvar.train(epochs=1)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model = self.deepvar.train(epochs=1)
        pred_mean, pred_std = self.deepvar.predict(model)
        self.assertTrue(pred_mean.shape == (self.n, self.s))

    def test_results_interval(self):
        model = self.deepvar.train(epochs=2)
        pred_mean, pred_std = self.deepvar.predict(model)
        res = self.deepvar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 25)
        self.assertLess(res['CRPS']['bottom_ind'][0], 200)

    def test_evaluate(self):
        hts_val = self.deepvar._build_validation_ds_multivariate()
        val_ds = hts_val.to_dataset()

        model = self.deepvar.train(epochs=2)

        mase_score = self.deepvar.evaluate(model, val_ds)

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=val_ds, predictor=model, num_samples=self.deepvar.n_samples
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        forecast_means = np.array([f.mean.squeeze() for f in forecasts]).squeeze()
        true_targets = np.array([ts.values for ts in tss]).squeeze()[-self.deepvar.h:]

        y_train = np.dot(self.deepvar.groups["train"]["data"], self.deepvar.S.T)
        naive_forecast_train = y_train[:-self.deepvar.seasonality]
        y_train_actual = y_train[self.deepvar.seasonality:]

        mae_naive_train = mean_absolute_error(y_train_actual, naive_forecast_train, multioutput="raw_values")
        mae_forecast = mean_absolute_error(true_targets, forecast_means, multioutput="raw_values")
        mase_adjusted = mae_forecast / mae_naive_train

        self.assertTrue(abs(mase_score.mean()-mase_adjusted.mean()) < 5)

    def test_hyper_tuning(self):
        self.deepvar.hyper_tuning(n_trials=2, epochs=2)


