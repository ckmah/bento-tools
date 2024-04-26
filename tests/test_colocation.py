import matplotlib
matplotlib.use("Agg")
import unittest

import spatialdata as sd
from unittest.mock import patch

import bento as bt
import os

class TestColocation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.imgdir = "/".join(bt.__file__.split("/")[:-2]) + "/tests/img/colocation"
        os.makedirs(self.imgdir, exist_ok=True)

        self.data = sd.read_zarr(f"{datadir}/merfish_sample.zarr")
        self.data = bt.io.prep(
            sdata=self.data,
            points_key="transcripts",
            feature_key="feature_name",
            instance_key="cell_boundaries",
            shape_keys=["cell_boundaries", "nucleus_boundaries"],
        )

        self.rank = 3

        bt.tl.coloc_quotient(self.data, shapes=["cell_boundaries"])
        bt.tl.colocation(self.data, ranks=range(1, self.rank + 1), plot_error=False)

    def test_coloc_quotient(self):
        # Check that clq is in self.data.table.uns
        self.assertTrue("clq" in self.data.table.uns)

        # Check that cell_boundaries is in self.data.table.uns["clq"]
        self.assertTrue("cell_boundaries" in self.data.table.uns["clq"])

        coloc_quotient_features = [
            "feature_name",
            "neighbor",
            "clq",
            "cell_boundaries",
            "log_clq",
            "compartment",
        ]
        # Check columns are in clq["cell_boundaries"]
        for feature in coloc_quotient_features:
            self.assertTrue(feature in self.data.table.uns["clq"]["cell_boundaries"])

        # Check that tensor is in self.data.table.uns
        self.assertTrue("tensor" in self.data.table.uns)

        # Check that tensor_labels is in self.data.table.uns
        self.assertTrue("tensor_labels" in self.data.table.uns)

        # Check that tensor_names is in self.data.table.uns
        self.assertTrue("tensor_names" in self.data.table.uns)

        # Check keys are in tensor_labels
        for feature in self.data.table.uns["tensor_names"]:
            self.assertTrue(feature in self.data.table.uns["tensor_labels"])

        # Check that factors is in self.data.table.uns
        self.assertTrue("factors" in self.data.table.uns)

        # Check that keys are in factors
        for i in range(1, 3):
            self.assertTrue(i in self.data.table.uns["factors"])

        # Check that factors_error is in self.data.table.uns
        self.assertTrue("factors_error" in self.data.table.uns)
        self.assertTrue("rmse" in self.data.table.uns["factors_error"])
        self.assertTrue("rank" in self.data.table.uns["factors_error"])

    @patch("matplotlib.pyplot.savefig")
    def test_colocation_plot(self, mock_savefig):
        bt.pl.colocation(
            self.data, rank=self.rank, fname=f"{self.imgdir}/colocation.png"
        )

        mock_savefig.assert_called()
