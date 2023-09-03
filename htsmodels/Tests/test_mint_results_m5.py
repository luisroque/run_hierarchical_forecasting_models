import unittest
import tsaugmentation as tsag
from htsmodels.models.mint import MinT


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = tsag.preprocessing.CreateGroups('m5', 'W').read_original_groups()
        self.n = self.data['predict']['n']
        self.s = self.data['train']['s']
        self.mint = MinT(dataset='m5', groups=self.data)

    def test_correct_train(self):
        model = self.mint.train()
        self.assertIsNotNone(model)


