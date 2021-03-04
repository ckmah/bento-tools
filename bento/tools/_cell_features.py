from shapely.geometry import Point

def cell_aspect_ratio(data, copy=False):

    adata = data.copy() if copy else data

    def _aspect_ratio(poly):
        # get coordinates of min bounding box vertices around polygon
        x, y = poly.minimum_rotated_rectangle.exterior.coords.xy

        # get length of bound box sides
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # length = longest side, width = shortest side
        length, width = max(edge_length), min(edge_length)

        # return long / short ratio
        return length / width

    ar = adata.obs['cell_shape'].apply(lambda poly: _aspect_ratio(poly))
    adata.obs['aspect-ratio'] = ar

    return adata if copy else None


def cell_area(data, copy=False):
    adata = data.copy() if copy else data

    # Calculate pixel-wise area 
    # TODO: unit scale?
    area = adata.obs['cell_shape'].area
    adata.obs['area'] = area

    return adata if copy else None
