import unittest
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.deepar = DeepAR(dataset='prison', groups=self.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_correct_train(self):
        model = self.deepar.train(epochs=10)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model = self.deepar.train(epochs=10)
        forecasts = self.deepar.predict(model)
        res = self.deepar.results(forecasts)
        self.assertTrue(res.shape == (self.n, self.s, 500))

    def test_results_interval(self):
        model = self.deepar.train(epochs=10)
        forecasts = self.deepar.predict(model)
        results = self.deepar.results(forecasts)
        res = self.deepar.metrics(results)
        self.assertLess(res['mase']['bottom'], 2.8)

    def test_wall_time(self):
        model = self.deepar.train(epochs=10)
        forecasts = self.deepar.predict(model)
        results = self.deepar.results(forecasts)
        res = self.deepar.metrics(results)
        self.assertLess(res['wall_time']['wall_time_total'], 50)
