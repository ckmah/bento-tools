from shapely.geometry import Point, Polygon, LineString
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import math
from ._sample_features import extract

def compare_subcellular_enrichment(data, cell_pop1, cell_pop2,genes=[],threshold=0.5,copy=False):
    adata = data.copy() if copy else data
    if type(cell_pop1) == list:
        pass
    else:
        cell_pop1 = [cell_pop1]
    if type(cell_pop2) == list:
        pass
    else:
        cell_pop2 = [cell_pop2]
    # get list of genes common to both populations of cells
    # gene is considered in a population if at least one cell in the population contains it
    df_points = adata.uns['points']
    df_pop1 = df_points[df_points['cell'].isin(cell_pop1)]
    df_pop2 = df_points[df_points['cell'].isin(cell_pop2)]
    genes_pop1 = list(np.unique(df_pop1['gene']))
    genes_pop2 = list(np.unique(df_pop2['gene']))
    common_genes = list(set(genes_pop1) & set(genes_pop2))
    # if user has already defined list of genes, use that. else use common genes
    if len(genes) == 0:
        comparison_genes = common_genes
    else:
        comparison_genes = [i for i in genes if i in common_genes]
    # average enrichment score for each gene per population
    pop1_enrich = {}
    pop2_enrich = {}
    for zone in adata.uns['zones']:
        pop1_enrich[zone] = {}
        pop2_enrich[zone] = {}
        zone_enrich_df = adata.to_df(zone+'_enrichment')[common_genes]
        pop1_enrich_df = zone_enrich_df.loc[cell_pop1]
        pop2_enrich_df = zone_enrich_df.loc[cell_pop2]
        for gene in comparison_genes:
            pop1_enrich[zone][gene] = pop1_enrich_df[gene].mean()
            pop2_enrich[zone][gene] = pop2_enrich_df[gene].mean()
    pop1_enr_df = pd.DataFrame.from_dict(pop1_enrich)
    pop2_enr_df = pd.DataFrame.from_dict(pop2_enrich)
    classification = []
    for gene,row in pop1_enr_df.iterrows():
        values = list(row.values)
        zones = list(row.index)
        best = max(values)
        if best < threshold:
            classification.append('None')
            continue
        else:
            classification.append(zones[values.index(best)])
    pop1_enr_df['enrichment'] = classification
    classification = []
    for gene,row in pop2_enr_df.iterrows():
        values = list(row.values)
        zones = list(row.index)
        best = max(values)
        if best < threshold:
            classification.append('None')
            continue
        else:
            classification.append(zones[values.index(best)])
    pop2_enr_df['enrichment'] = classification
    # precompute simplex plot coordinates for future use
    pop1_enr_df = _unit_vec_assign(pop1_enr_df,adata.uns['zones'])
    pop2_enr_df = _unit_vec_assign(pop2_enr_df,adata.uns['zones'])
     # combine both dataframes into one
    simplex_df = pd.DataFrame(index=pop1_enr_df.index)
    simplex_df['population1_class'] = pop1_enr_df['enrichment']
    simplex_df['population2_class'] = pop2_enr_df['enrichment']
    simplex_df['population1_simplex_x'] = pop1_enr_df['plot_x']
    simplex_df['population1_simplex_y'] = pop1_enr_df['plot_y']
    simplex_df['population2_simplex_x'] = pop2_enr_df['plot_x']
    simplex_df['population2_simplex_y'] = pop2_enr_df['plot_y']
    # filter out genes that classify as "None" in any of the two populations
    simplex_df = simplex_df[simplex_df['population1_class'] != 'None']
    simplex_df = simplex_df[simplex_df['population2_class'] != 'None']
    # save values in adata.uns
    adata.uns['enrichment_comparison'] = {}
    adata.uns['enrichment_comparison']['cell_population1'] = cell_pop1
    adata.uns['enrichment_comparison']['cell_population2'] = cell_pop2
    adata.uns['enrichment_comparison']['genes'] = list(simplex_df.index)
    adata.uns['enrichment_comparison']['simplex_summary'] = simplex_df
    adata.uns['enrichment_comparison']['population1_enrichment'] = pop1_enr_df
    adata.uns['enrichment_comparison']['population2_enrichment'] = pop2_enr_df
    return adata if copy else None

def _unit_vec_assign(enrichment_data,classes):
    # enrichment_data is a pandas dataframe with genes as index and enrichment classes as columns
    # data must be normalized to a 0-1 scale
    zone_vecs = _unit_vectors(len(classes))
    plot_x = []
    plot_y = []
    for gene, row in enrichment_data.iterrows():
        x = 0
        y = 0
        n = 0
        for cl in classes:
            x = x + (zone_vecs[n][0] * row[cl])
            y = y + (zone_vecs[n][1] * row[cl])
            n += 1
        plot_x.append(x)
        plot_y.append(y)
    enrichment_data['plot_x'] = plot_x
    enrichment_data['plot_y'] = plot_y
    return enrichment_data

def _unit_vectors(num_zones):
    vectors = []
    for i in range(num_zones):
        rad = (math.pi*(4*i + num_zones))/(2*num_zones)
        x = round(math.cos(rad),2)
        y = round(math.sin(rad),2)
        vectors.append((x,y))
    return vectors


def define_zones(data, boundary_zones=['nucleus_shape','cell_shape'],priority=['nucleus_shape','cell_shape'],proportions={'cell_shape':(0,-0.05)},copy=False):
    # proportions if given are in format (+/-)
    adata = data.copy() if copy else data

    # quick checks
    shape_cols = [i for i in adata.obs.columns if 'shape' in i]
    assert len(priority) == len(shape_cols)
    for zone in boundary_zones:
        assert zone in priority
    # apply to each cell
    # Create dictionary of lists to hold results
    all_zones = {}
    for z in priority:
        all_zones[z+'_zone'] = []
    for z in boundary_zones:
        all_zones[z+'_edge_zone'] = []
    print("Computing all subcellular compartments and boundaries...")
    for c, cell_obs in tqdm(adata.obs.iterrows(),total=len(adata.obs)):
        zones = _overlay_zones(cell_obs,boundary_zones,priority,proportions)
        for key,val in zones.items():
            all_zones[key].append(val)
    # add to adata
    zone_list = []
    for z,column in all_zones.items():
        adata.obs[z] = column
        zone_list.append(z)
    adata.uns['zones'] = zone_list

    return adata if copy else None

def _overlay_zones(cell_obs,boundary_zones,priority,proportions):
    zones = {}
    for mask in priority:
        if mask in boundary_zones:
            if mask in proportions:
                proportion = proportions[mask]
            else:
                proportion = (0.05,-0.05)
            boundary_zone = _create_boundary_zone(cell_obs[mask],proportion)
            inner_zone = cell_obs[mask].difference(boundary_zone)
            # subtract other zones from these
            for z, poly in zones.items():
                boundary_zone = boundary_zone.difference(poly)
                inner_zone = inner_zone.difference(poly)
            zones[mask+'_zone'] = inner_zone
            zones[mask+'_edge_zone'] = boundary_zone
        else:
            inner_zone = cell_obs[mask]
            for z, poly in zones.items():
                inner_zone = inner_zone.difference(poly)
            zones[mask+'_zone'] = inner_zone
    return zones


def _create_boundary_zone(mask,proportion):
    # create a symmetrical +/- buffer proportional to size of the geometry fo the mask
    # first create minimum bounding box --> get minor axis
    mbr_points = list(zip(*mask.minimum_rotated_rectangle.exterior.coords.xy))
    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]
    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    buff_plus = proportion[0]*minor_axis
    buff_minus = proportion[1]*minor_axis
    boundary_zone = mask.buffer(buff_plus).difference(mask.buffer(buff_minus))
    return boundary_zone

