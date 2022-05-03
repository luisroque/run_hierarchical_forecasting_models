import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('m5', test_size=2).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.mint = MinT(dataset='m5', groups=self.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        forecasts = self.mint.train()
        res = self.mint.results(forecasts)
        self.assertTrue(res.shape == (576, 12))
