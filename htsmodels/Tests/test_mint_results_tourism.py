import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228*10).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.mint = MinT(dataset='tourism', groups=self.data,
                         store_prediction_samples=True,
                         store_prediction_points=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_results_interval(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        res = self.mint.metrics(results)
        self.mint.store_metrics(res)
        self.assertLess(res['mase']['bottom'], 2.7)
        self.assertLess(res['CRPS']['bottom_ind'][0], 10)

    def test_predict_shape(self):
        forecasts = self.mint.train()
        df_results = self.mint.results(forecasts)
        self.assertTrue(df_results.shape == (576, 11))
