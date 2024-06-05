import pytest


import bento as bt
from . import conftest


@pytest.fixture(scope="module")
def flux_data(small_data):
    bt.tl.flux(
        sdata=small_data,
        points_key="transcripts",
        instance_key="cell_boundaries",
        feature_key="feature_name",
        res=conftest.FLUX_RES,
        radius=conftest.FLUX_RADIUS,
    )

    return small_data


def test_flux(flux_data):
    # Check that cell_boundaries_raster is in flux_data.points
    assert "cell_boundaries_raster" in flux_data.points

    # Check that flux_genes is in flux_data.table.uns
    assert "flux_genes" in flux_data.table.uns
    genes = flux_data.table.uns["flux_genes"]

    # Check that flux_variance_ratio is in flux_data.table.uns
    assert "flux_variance_ratio" in flux_data.table.uns

    # Check columns are added in cell_boundaries_raster
    assert all(
        gene in flux_data.points["cell_boundaries_raster"].columns for gene in genes
    )

    for i in range(len(genes)):
        assert f"flux_embed_{i}" in flux_data.points["cell_boundaries_raster"].columns


def test_fluxmap(flux_data):
    # May fail if coordinates are negative
    bt.tl.fluxmap(
        sdata=flux_data,
        points_key="transcripts",
        instance_key="cell_boundaries",
        res=conftest.FLUX_RES,
        min_count=conftest.FLUXMAP_MIN_COUNT,
        train_size=conftest.FLUXMAP_TRAIN_SIZE,
        n_clusters=conftest.FLUXMAP_N_CLUSTERS,
        plot_error=False,
    )
    assert "fluxmap" in flux_data.points["cell_boundaries_raster"].columns
    for i in range(1, conftest.FLUXMAP_N_CLUSTERS + 1):
        assert f"fluxmap{i}" in flux_data.points["transcripts"].columns
        assert f"fluxmap{i}" in flux_data.shapes


# @patch("matplotlib.pyplot.savefig")
# def test_flux_plot(self, mock_savefig):
#     bt.pl.flux(flux_data, res=1, fname=f"{self.imgdir}/flux.png")


def test_fe_fazal2019(flux_data):
    bt.tl.fe_fazal2019(flux_data)

    # Check that cell_boundaries_raster is in flux_data.points
    assert "cell_boundaries_raster" in flux_data.points

    # Check that fe_stats is in flux_data.table.uns
    assert "fe_stats" in flux_data.table.uns

    # Check that fe_ngenes is in flux_data.table.uns
    assert "fe_ngenes" in flux_data.table.uns

    # Check columns are in cell_boundaries_raster, fe_stats, abd fe_ngenes
    for feature in conftest.FAZAL2019_FEATURES:
        assert f"flux_{feature}" in flux_data.points["cell_boundaries_raster"]
        assert feature in flux_data.table.uns["fe_stats"]
        assert feature in flux_data.table.uns["fe_ngenes"]


def test_fe_xia2019(flux_data):
    bt.tl.fe_xia2019(flux_data)

    # Check that cell_boundaries_raster is in flux_data.points
    assert "cell_boundaries_raster" in flux_data.points

    # Check that fe_stats is in flux_data.table.uns
    assert "fe_stats" in flux_data.table.uns

    # Check that fe_ngenes is in flux_data.table.uns
    assert "fe_ngenes" in flux_data.table.uns

    # Check columns are in cell_boundaries_raster, fe_stats, abd fe_ngenes
    for feature in conftest.XIA2019_FEATURES:
        assert f"flux_{feature}" in flux_data.points["cell_boundaries_raster"]
        assert feature in flux_data.table.uns["fe_stats"]
        assert feature in flux_data.table.uns["fe_ngenes"]


# def test_fluxmap_plot(flux_data):
#     bt.tl.fluxmap(
#         sdata=flux_data,
#         points_key="transcripts",
#         instance_key="cell_boundaries",
#         res=self.res,
#         train_size=1,
#         n_clusters=3,
#         plot_error=False,
#     )
#     plt.figure()
#     bt.pl.fluxmap(flux_data, fname=f"{self.imgdir}/fluxmap.png")

# def test_fe_plot(flux_data):
#     # TODO this test is so slow
#     bt.tl.fluxmap(
#         sdata=flux_data,
#         points_key="transcripts",
#         instance_key="cell_boundaries",
#         res=self.res,
#         train_size=1,
#         n_clusters=3,
#         plot_error=False,
#     )
#     bt.tl.fe_fazal2019(flux_data)

#     plt.figure()
#     bt.pl.fe(
#         flux_data,
#         "flux_OMM",
#         res=self.res,
#         shapes=["cell_boundaries", "fluxmap1_boundaries"],
#         fname=f"{self.imgdir}/fe_flux_OMM_fluxmap1.png",
#     )
