import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228*2).apply_preprocess()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        shutil.rmtree("./data/original_datasets")
        self.mint = MinT(dataset='tourism', groups=self.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)

    def test_predict_shape(self):
        forecasts = self.mint.train()
        df_results = self.mint.results(forecasts)
        self.assertTrue(df_results.shape == (576, 11))
