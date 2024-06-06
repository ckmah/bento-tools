import bento as bt
import pandas as pd


def test_comp(small_data):
    # Set up test data
    shape_names = ["cell_boundaries", "nucleus_boundaries"]
    points_key = "transcripts"

    # Call the comp function
    bt.tl.comp(sdata=small_data, points_key=points_key, shape_names=shape_names)

    # Check if comp_stats is updated in sdata.table.uns
    assert "comp_stats" in small_data.table.uns

    # Check the type of comp_stats
    assert type(small_data.table.uns["comp_stats"]) == pd.DataFrame

    # Check if the shape_names are present in comp_stats
    assert all(
        shape_name in small_data.table.uns["comp_stats"] for shape_name in shape_names
    )
