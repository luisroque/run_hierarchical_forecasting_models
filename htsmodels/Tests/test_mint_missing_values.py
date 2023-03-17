import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
from htsmodels.preprocessing.subsampled_dataset import update_missing_values


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.CreateGroups('prison', 0.75).read_subsampled_groups()
        self.data_orig = tsag.preprocessing.CreateGroups('prison').read_original_groups()
        self.data_w_missing_values = update_missing_values(self.data)
        self.data_w_missing_values['predict'] = self.data_orig['predict']
        self.mint = MinT(dataset='prison', groups=self.data_w_missing_values)

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_results_interval(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        pred_mean, pred_std = self.mint.predict(results)
        res = self.mint.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.2)
        self.assertLess(res['CRPS']['bottom_ind'][0], 100)

