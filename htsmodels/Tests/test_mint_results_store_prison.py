import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.mint = MinT(dataset='prison', groups=self.data, input_dir='./results/mint/')

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_results_interval(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        res = self.mint.metrics(results)
        self.mint.store_metrics(res)
        self.assertLess(res['mase']['bottom'], 2.2)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_wall_time(self):
        forecasts = self.mint.train()
        df_results = self.mint.results(forecasts)
        res = self.mint.metrics(df_results)
        self.assertLess(res['wall_time']['wall_time_total'], 50)
