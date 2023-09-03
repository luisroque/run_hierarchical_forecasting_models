import unittest
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
from htsmodels.preprocessing.subsampled_dataset import update_missing_values


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.CreateGroups('prison', 'Q', 0.5).read_subsampled_groups()
        self.data_orig = tsag.preprocessing.CreateGroups('prison', 'Q').read_original_groups()
        self.data_w_missing_values = update_missing_values(self.data, 'QS')
        self.data_w_missing_values['predict'] = self.data_orig['predict']
        self.deepar = DeepAR(dataset='prison', groups=self.data_w_missing_values)

    def test_correct_train(self):
        model = self.deepar.train(epochs=5)
        self.assertIsNotNone(model)

    def test_results_interval(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 3)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

