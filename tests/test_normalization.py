import unittest
import numpy as np
import numpy.testing as npt
from pbdl.loader import *
import tests.setup
import h5py


class TestNormalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        tests.setup.setup()
        self.rand_dset = h5py.File("tests/datasets/random.hdf5", "r+")

    def tearDown(self):
        self.rand_dset.close()

    def test_std_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            10,
            normalize=StdNorm(),
            batch_size=1,
        )

        std_input = [0] * 3  # 3 fields (1 vector field, 2 scalar fields)
        std_target = [0] * 3

        for input, target in loader:
            input = input[0]
            target = target[0]

            # calculate vector norm for velocity
            input[0] = np.linalg.norm(input[0:2], axis=0)
            input = np.delete(input, 1, axis=0)

            # calculate vector norm for velocity
            target[0] = np.linalg.norm(target[0:2], axis=0)
            target = np.delete(target, 1, axis=0)

            # calculating std over spatial dims
            for f in range(3):
                std_input[f] += np.std(input[f])
                std_target[f] += np.std(input[f])

        for f in range(3):
            std_input[f] /= len(loader)
            std_target[f] /= len(loader)

            self.assertAlmostEqual(std_input[f], 1, places=2)
            self.assertAlmostEqual(std_input[f], 1, places=2)

    def test_std_norm_rev(self):
        norm = StdNorm()
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize_data(arr)
        arr_rev = norm.normalize_data_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)

    def test_mean_std_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            1,
            normalize=MeanStdNorm(),
            batch_size=1,
        )

        mean_input = [0] * 4  # 4 scalar fields
        std_input = [0] * 4

        mean_target = [0] * 4
        std_target = [0] * 4

        for input, target in loader:
            input = input[0]
            target = target[0]

            for sf in range(4):
                mean_input[sf] += np.sum(input[sf])
                mean_target[sf] += np.sum(target[sf])
                std_input[sf] += np.std(input[sf])
                std_target[sf] += np.std(target[sf])

        for sf in range(4):
            mean_input[sf] /= len(loader)
            mean_target[sf] /= len(loader)
            std_input[sf] /= len(loader)
            std_target[sf] /= len(loader)

            self.assertAlmostEqual(
                mean_input[sf], 0, places=0
            )  # TODO precision, dataset too small
            self.assertAlmostEqual(mean_target[sf], 0, places=0)
            self.assertAlmostEqual(std_input[sf], 1, places=2)
            self.assertAlmostEqual(std_target[sf], 1, places=2)

    def test_mean_std_norm_rev(self):
        norm = MeanStdNorm()
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize_data(arr)
        arr_rev = norm.normalize_data_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)

    def test_min_max_norm(self):
        loader = Dataloader(
            "transonic-cylinder-flow-tiny",
            10,
            normalize=MinMaxNorm(min_val=-1, max_val=1),
            batch_size=1,
        )

        min_input = [float("inf")] * 4  # 4 scalar fields
        max_input = [-float("inf")] * 4
        min_target = [float("inf")] * 4
        max_target = [-float("inf")] * 4

        for input, target in loader:
            input = input[0]
            target = target[0]

            for sf in range(4):
                min_input[sf] = min(min_input[sf], np.min(input[sf]))
                max_input[sf] = max(max_input[sf], np.max(input[sf]))
                min_target[sf] = min(min_target[sf], np.min(target[sf]))
                max_target[sf] = max(max_target[sf], np.max(target[sf]))

        for min_val in min_input:
            self.assertAlmostEqual(min_val, -1, places=5)

        for max_val in max_input:
            self.assertAlmostEqual(max_val, 1, places=5)

        for min_val in min_target:
            self.assertAlmostEqual(min_val, -1, places=5)

        for max_val in max_target:
            self.assertAlmostEqual(max_val, 1, places=5)

    def test_min_max_norm_rev(self):
        norm = MinMaxNorm(-1, 1)
        norm.prepare(self.rand_dset, sel_const=None)

        arr = np.random.rand(4, 128, 64)

        arr_norm = norm.normalize_data(arr)
        arr_rev = norm.normalize_data_rev(arr_norm)

        npt.assert_array_almost_equal(arr_rev, arr)
