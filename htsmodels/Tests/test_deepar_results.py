import unittest
import pickle
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
from htsmodels import __version__


class TestModel(unittest.TestCase):

    def setUp(self):
        self.dataset_name = 'prison'
        self.data = tsag.preprocessing.PreprocessDatasets(self.dataset_name).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['predict']['s']
        self.h = self.data['h']
        self.version = __version__
        self.deepar = DeepAR(dataset=self.dataset_name, groups=self.data)

    def test_correct_train(self):
        model = self.deepar.train(epochs=5)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        self.assertTrue(pred_mean.shape == (self.n, self.s))

    def test_results_interval(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.8)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_results_interval_negbindist(self):
        model = self.deepar.train(epochs=5, dist='NegativeBinomial')
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.8)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_store_pred_std_results_gp(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        res_type = "pred"
        res_measure = "std"
        self.deepar.store_results(
            pred_std[self.n - self.h :, :], res_type=res_type, res_measure=res_measure
        )
        with open(
            f"{self.deepar.input_dir}results_{res_type}_{res_measure}_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        self.assertTrue(res.shape == (self.h, self.s))

    def test_store_metrics_mint(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        self.deepar.input_dir = f"./results/deepar/"
        res = self.deepar.metrics(pred_mean, pred_std)
        self.deepar.store_metrics(res)
        with open(
            f"{self.deepar.input_dir}metrics_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        keys = list(res.keys())
        self.assertTrue(keys == ['mase', 'rmse', 'CRPS', 'wall_time'])

