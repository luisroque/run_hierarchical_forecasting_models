import unittest
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison').apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['predict']['s']
        self.h = self.data['h']
        shutil.rmtree("./data/original_datasets")
        self.deepar = DeepAR(dataset='prison', groups=self.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_correct_train(self):
        model = self.deepar.train(epochs=5)
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        self.assertTrue(pred_mean.shape == (self.n, self.s))

    def test_results_interval(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.8)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_results_interval_negbindist(self):
        model = self.deepar.train(epochs=5, dist='NegativeBinomial')
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['mase']['bottom'], 2.8)
        self.assertLess(res['CRPS']['bottom_ind'][0], 5)

    def test_wall_time(self):
        model = self.deepar.train(epochs=5)
        pred_mean, pred_std = self.deepar.predict(model)
        res = self.deepar.metrics(pred_mean, pred_std)
        self.assertLess(res['wall_time']['wall_time_total'], 50)
