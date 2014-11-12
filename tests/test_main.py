import unittest
import numpy as np
from main import class_accuracy, dual_shuffle
from copy import deepcopy
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

    def test_dual_shuffle(self):
        a = ['Spears', "Adele", "NDubz", "Nicole", "Cristina"]
        a_copy = deepcopy(a)
        b = [1, 2, 3, 4, 5]
        b_copy = deepcopy(b)
        self.assertEqual(a, a_copy)
        self.assertEqual(b, b_copy)
        dual_shuffle(a, b)
        self.assertNotEqual(a, a_copy)
        self.assertNotEqual(b, b_copy)
        self.assertEqual(b_copy, sorted(b))
        indices = np.argsort(b)
        a_sorted = list(np.array(a)[indices])
        self.assertEqual(a_sorted, a_copy)


if __name__ == '__main__':
    unittest.main()