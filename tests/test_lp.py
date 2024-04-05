import unittest
import bento as bt
import spatialdata as sd
import random


class TestLp(unittest.TestCase):
    def setUp(self):
        datadir = "/".join(bt.__file__.split("/")[:-1]) + "/datasets"
        self.data = sd.read_zarr(f"{datadir}/bt_small_data.zarr")

        bt.tl.lp(
            sdata=self.data,
            instance_key="cell_boundaries",
            nucleus_key="nucleus_boundaries",
            groupby="feature_name",
        )
        bt.tl.lp_stats(sdata=self.data, instance_key="cell_boundaries")

    def test_lp(self):
        lp_columns = [
            "cell_boundaries",
            "feature_name",
            "cell_edge",
            "cytoplasmic",
            "none",
            "nuclear",
            "nuclear_edge",
        ]

        # Check lp and lpp dataframes in sdata.table.uns
        for column in lp_columns:
            self.assertTrue(column in self.data.table.uns["lp"].columns)
            self.assertTrue(column in self.data.table.uns["lpp"].columns)

    def test_lp_stats(self):
        lp_stats_columns = [
            "cell_edge",
            "cytoplasmic",
            "none",
            "nuclear",
            "nuclear_edge",
        ]

        # Check lp_stats index in sdata.table.uns
        self.assertTrue(self.data.table.uns["lp_stats"].index.name == "feature_name")
        # Check lp_stats dataframe in sdata.table.uns
        for column in lp_stats_columns:
            self.assertTrue(column in self.data.table.uns["lp_stats"].columns)

    def test_lp_diff_discrete(self):
        lp_diff_discrete_columns = [
            "feature_name",
            "pattern",
            "phenotype",
            "dy/dx",
            "std_err",
            "z",
            "pvalue",
            "ci_low",
            "ci_high",
            "padj",
            "-log10p",
            "-log10padj",
            "log2fc",
        ]

        # Assign random cell stage to each cell
        stages = ["G0", "G1", "S", "G2", "M"]
        phenotype = []
        for i in range(len(self.data.shapes["cell_boundaries"])):
            phenotype.append(random.choice(stages))
        self.data.shapes["cell_boundaries"]["cell_stage"] = phenotype

        bt.tl.lp_diff_discrete(
            sdata=self.data, instance_key="cell_boundaries", phenotype="cell_stage"
        )

        # Check lp_diff_discrete dataframe in sdata.table.uns
        for column in lp_diff_discrete_columns:
            self.assertTrue(column in self.data.table.uns["diff_cell_stage"].columns)

    def test_lp_diff_discrete_error(self):
        error_test = []
        for i in range(len(self.data.shapes["cell_boundaries"])):
            if (
                self.data.shapes["cell_boundaries"]["cell_boundaries_area"][i]
                > self.data.shapes["cell_boundaries"]["cell_boundaries_area"].median()
            ):
                error_test.append(1)
            else:
                error_test.append(0)
        self.data.shapes["cell_boundaries"]["error_test"] = error_test

        # Check that KeyError is raised when phenotype is numeric
        with self.assertRaises(KeyError):
            bt.tl.lp_diff_discrete(
                sdata=self.data, instance_key="cell_boundaries", phenotype="error_test"
            )

    def test_lp_diff_continuous(self):
        lp_diff_continuous_columns = ["feature_name", "pattern", "pearson_correlation"]

        bt.tl.lp_diff_continuous(self.data, phenotype="cell_boundaries_area")

        # Check lp_diff_continuous dataframe in sdata.table.uns
        for column in lp_diff_continuous_columns:
            self.assertTrue(
                column in self.data.table.uns["diff_cell_boundaries_area"].columns
            )
