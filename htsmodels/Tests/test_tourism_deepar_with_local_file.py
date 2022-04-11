import unittest
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.preproc = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228*10)
        self.data = self.preproc._tourism()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.deepar = DeepAR(dataset='tourism', groups=self.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_results_mean_and_prediction_interval(self):
        model = self.deepar.train(epochs=10)
        forecasts = self.deepar.predict(model)
        mean, lower, upper = self.deepar.results(forecasts)
        res = self.deepar.metrics(mean, lower, upper)
        self.assertTrue(res['mean'].shape == (1, self.n, self.s))
        self.assertTrue(res['lower'].shape == (1, self.n, self.s))
        self.assertTrue(res['upper'].shape == (1, self.n, self.s))