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
        model = self.deepar.train(epochs=2)
        forecasts = self.deepar.predict(model)
        samples = self.deepar.results(forecasts)
        res = self.deepar.metrics(samples)

        # Test shape of results
        self.assertTrue(res['mase']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['CRPS']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['rmse']['bottom_ind'].shape == (self.s, ))

        # Test shape of predictions
        self.assertTrue(res['predictions']['samples']['bottom'].shape == (self.n, self.s, 500))
