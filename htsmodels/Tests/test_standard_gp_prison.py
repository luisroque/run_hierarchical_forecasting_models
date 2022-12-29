import unittest
import tsaugmentation as tsag
from htsmodels.models.standard_gp import SGP


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.gpf = SGP('prison', self.data)

    def test_results_interval_piecewise(self):
        model, like = self.gpf.train(n_iterations=100)
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

    def test_results_interval_linear(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='linear')
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

    def test_results_interval_zero(self):
        model, like = self.gpf.train(n_iterations=100, mean_function='zero')
        pred, pred_scaled = self.gpf.predict(model, like)
        res = self.gpf.metrics(pred[0], pred[1])
        self.assertLess(res['mase']['bottom'], 5)

