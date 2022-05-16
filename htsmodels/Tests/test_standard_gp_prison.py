import unittest
from htsmodels.models.standard_gp import SGP
import tsaugmentation as tsag
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.gpf = SGP('prison', self.data)

    def test_results_interval_piecewise(self):
        model, like = self.gpf.train(n_iterations=100)
        samples = self.gpf.predict(model, like)
        res = self.gpf.metrics(samples)
        self.assertLess(res['mase']['bottom'], 2.5)

    def test_results_interval_linear(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='linear')
        samples = self.gpf.predict(model, like)
        res = self.gpf.metrics(samples)
        self.assertLess(res['mase']['bottom'], 2.5)
