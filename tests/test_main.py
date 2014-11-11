import unittest
import numpy as np
from main import class_accuracy
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        print 'Setup complete'

    def test_accuracy(self):
        labels_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        labels_pred = np.array([0, 0, 1, 0, 3, 3, 3, 1, 3, 2])
        accuracies = class_accuracy(labels_true, labels_pred)
        correct = {
            0: 1.0,
            1: 0.5,
            2: 0.0,
            3: 0.5,
        }
        self.assertEqual(accuracies, correct)

if __name__ == '__main__':
    unittest.main()