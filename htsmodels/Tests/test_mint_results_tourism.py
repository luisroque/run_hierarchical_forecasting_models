import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('tourism', 'M', test_size=228*10).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.mint = MinT(dataset='tourism', groups=self.data)

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
