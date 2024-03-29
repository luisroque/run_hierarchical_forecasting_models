import unittest
import pickle
import tsaugmentation as tsag
from htsmodels.models.gluon_models import TFT
from htsmodels import __version__


class TestModel(unittest.TestCase):

    def setUp(self):
        self.dataset_name = 'prison'
        self.data = tsag.preprocessing.PreprocessDatasets(self.dataset_name, 'Q').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['predict']['s']
        self.h = self.data['h']
        self.version = __version__
        self.tft = TFT(dataset=self.dataset_name, groups=self.data)

    def test_correct_train(self):
        model = self.tft.train(epochs=1)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model = self.tft.train(epochs=1)
        pred_mean, pred_std = self.tft.predict(model)
        self.assertTrue(pred_mean.shape == (self.n, self.s))

    def test_results_interval(self):
        model = self.tft.train(epochs=20)
        pred_mean, pred_std = self.tft.predict(model)
        res = self.tft.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 22)
        self.assertLess(res['CRPS']['bottom_ind'][0], 120)

