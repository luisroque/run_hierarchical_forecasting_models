import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT
import shutil


class TestModel(unittest.TestCase):

    def setUp(self):
        self.preproc = tsag.preprocessing.PreprocessDatasets('prison')
        self.data = self.preproc._prison()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.mint = MinT(dataset='tourism', groups=self.data,
                         store_prediction_samples=True,
                         store_prediction_points=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./results")

    def test_results_mean_and_prediction_interval(self):
        forecasts = self.mint.train()
        df_results = self.mint.results(forecasts)
        res = self.mint.metrics(df_results)

        # Test shape of results
        self.assertTrue(res['mase']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['CRPS']['bottom_ind'].shape == (self.s, ))
        self.assertTrue(res['rmse']['bottom_ind'].shape == (self.s, ))

        # Test shape of predictions
        # Number of prediction samples for the bottom time series (n_points, n_series, n_samples)
        self.assertTrue(res['predictions']['samples']['bottom'].shape == (self.data['h'], self.s, 500))
        self.assertTrue(res['predictions']['points']['bottom'].shape == (self.data['h'], self.s))
        # Number of prediction samples for the total time series
        self.assertTrue(res['predictions']['samples']['total'].shape == (self.data['h'], 500))
        self.assertTrue(res['predictions']['points']['total'].shape == (self.data['h'],))
        # Test number of objects predicted and stored
        self.assertTrue(len(res['predictions']['samples']) == 14)
        self.assertTrue(len(res['predictions']['points']) == 14)
