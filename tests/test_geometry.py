import bento as bt
import geopandas as gpd


def test_overlay_intersection(small_data):
    s1 = "nucleus_boundaries"
    s2 = "cell_boundaries"
    name = "overlay_result"

    # Perform overlay operation using GeoDataFrame.overlay()
    shape1 = small_data.shapes[s1]
    shape2 = small_data.shapes[s2]
    expected_result = shape1.overlay(shape2, how="intersection", make_valid=True)

    # Perform overlay operation using bento.geo.overlay()
    bt.geo.overlay(small_data, s1, s2, name, how="intersection")

    assert name in small_data.shapes
    assert isinstance(small_data.shapes[name], gpd.GeoDataFrame)
    assert (
        small_data[name]
        .geom_equals_exact(expected_result, decimal=1, align=False)
        .all()
    )
