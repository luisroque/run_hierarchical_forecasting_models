import unittest
import pickle
import tsaugmentation as tsag
from htsmodels.models.standard_gp import SGP
from htsmodels import __version__


class TestModel(unittest.TestCase):

    def setUp(self):
        self.dataset_name = 'prison'
        self.data = tsag.preprocessing.PreprocessDatasets(self.dataset_name, 'Q').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.h = self.data['h']
        self.version = __version__
        self.gpf = SGP('prison', self.data)

    def test_results_interval_piecewise(self):
        model, like = self.gpf.train(n_iterations=100)
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

    def test_results_interval_linear(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='linear')
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

    def test_results_interval_zero(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='zero')
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

    def test_store_pred_std_results_gp(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='zero')
        preds, preds_scaled = self.gpf.predict(model, like)
        res_type = "pred"
        res_measure = "std"
        self.gpf.store_results(
            preds[1][self.n - self.h :, :], res_type=res_type, res_measure=res_measure
        )
        with open(
            f"{self.gpf.input_dir}results_{res_type}_{res_measure}_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        self.assertTrue(res.shape == (self.h, self.s))

    def test_store_metrics_gp(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='zero')
        pred, pred_scaled = self.gpf.predict(model, like)
        self.gpf.input_dir = f"./results/gpf/"
        preds, preds_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(preds[0], preds[1])
        self.gpf.store_metrics(res)
        with open(
            f"{self.gpf.input_dir}metrics_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        keys = list(res.keys())
        self.assertTrue(keys == ['mase', 'rmse', 'CRPS', 'wall_time'])
