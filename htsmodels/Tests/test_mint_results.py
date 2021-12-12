import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        self.mint = MinT(dataset='prison', groups=self.data)

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        forecasts = self.mint.train()
        res = self.mint.results(forecasts)
        self.assertTrue(res.shape == (1, self.n, self.s))

    def test_results_interval(self):
        forecasts = self.mint.train()
        results = self.mint.results(forecasts)
        res = self.mint.metrics(results)
        self.assertLess(res['mase']['bottom'], 2.2)
