import unittest
import pickle
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
from htsmodels import __version__


class TestModel(unittest.TestCase):

    def setUp(self):
        self.dataset_name = "prison"
        self.data = tsag.preprocessing.PreprocessDatasets(self.dataset_name).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.h = self.data['h']
        self.version = __version__
        self.mint = MinT(dataset=self.dataset_name, groups=self.data, input_dir='./results/mint/')
        self.mint.input_dir = f"./results/mint/"

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_results_interval(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        pred_mean, pred_std = self.mint.predict(results)
        res = self.mint.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.2)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_store_pred_std_results_gp(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        pred_mean, pred_std = self.mint.predict(results)
        res_type = "pred"
        res_measure = "std"
        self.mint.store_results(
            pred_std[self.n - self.h :, :], res_type=res_type, res_measure=res_measure
        )
        with open(
            f"{self.mint.input_dir}results_{res_type}_{res_measure}_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        self.assertTrue(res.shape == (self.h, self.s))

    def test_store_metrics_mint(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        pred_mean, pred_std = self.mint.predict(results)
        res = self.mint.metrics(pred_mean, pred_std)
        self.mint.store_metrics(res)
        with open(
            f"{self.mint.input_dir}metrics_gp_cov_{self.dataset_name}_{self.version}.pickle",
            "rb",
        ) as handle:
            res = pickle.load(handle)
        keys = list(res.keys())
        self.assertTrue(keys == ['mase', 'rmse', 'CRPS', 'wall_time'])
