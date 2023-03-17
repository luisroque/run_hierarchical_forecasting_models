import unittest
import tsaugmentation as tsag
from htsmodels.models.deepar import DeepAR
from htsmodels.preprocessing.subsampled_dataset import update_missing_values


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.CreateGroups('prison', 0.75).read_subsampled_groups()
        self.data_w_missing_values = update_missing_values(self.data)
        self.deepar = DeepAR(dataset='prison', groups=self.data_w_missing_values)

    def test_correct_train(self):
        model = self.deepar.train(epochs=5)
        self.assertIsNotNone(model)

