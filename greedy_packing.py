import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import os
import glob
import os.path

# Configuration
# put the path to the precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp'

# in order to ruin this to pack GPO votes, use the flag --maximize gop
# otherwise, it defaults to dem or use --maximize dem

SHAPEFILE_PATH = '/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp'
POP_TOLERANCE = 0.05
NUM_DISTRICTS = 17

def compute_votes(gdf):
    mapping = {
        '1st': ('GCON01DJAC','GCON01RCAR'),'2nd': ('GCON02DKEL','GCON02RLYN'),
        '3rd': ('GCON03DRAM','GCON03RBUR'),'4th': ('GCON04DGAR','GCON04RFAL'),
        '5th': ('GCON05DQUI','GCON05RHAN'),'6th': ('GCON06DCAS','GCON06RPEK'),
        '7th': ('GCON07DDAV','GCON07OWRI'),'8th': ('GCON08DKRI','GCON08RDAR'),
        '9th': ('GCON09DSCH','GCON09RRIC'),'10th': ('GCON10DSCH','GCON10RSEV'),
        '11th': ('GCON11DFOS','GCON11RLAU'),'12th': ('GCON12DMAR','GCON12RBOS'),
        '13th': ('GCON13DBUD','GCON13RDEE'),'14th': ('GCON14DUND','GCON14RGRY'),
        '15th': ('GCON15DLAN','GCON15RMIL'),'16th': ('GCON16DHAD','GCON16RLAH'),
        '17th': ('GCON17DSOR','GCON17RKIN')
    }
    gdf['Democratic_votes'] = 0
    gdf['Republican_votes'] = 0
    for dv, rv in mapping.values():
        gdf['Democratic_votes'] += gdf[dv].fillna(0)
        gdf['Republican_votes'] += gdf[rv].fillna(0)
    gdf['precinct_pop'] = gdf['Democratic_votes'] + gdf['Republican_votes']
    return gdf

# builds adjacency graph of precincts
def create_adjacency_graph(gdf):
    gdf = gdf.reset_index(drop=True)
    sindex = gdf.sindex
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    for i, geom in enumerate(gdf.geometry):
        for j in sindex.intersection(geom.bounds):
            if i >= j:
                continue
            if geom.intersects(gdf.geometry[j]):
                G.add_edge(i, j)
    return G

def greedy_logic(gdf, G, maximize='dem'):
    total_pop = gdf['precinct_pop'].sum()
    target = total_pop / NUM_DISTRICTS
    min_pop = target * (1 - POP_TOLERANCE)
    max_pop = target * (1 + POP_TOLERANCE)

    valid = set(gdf.index[gdf['precinct_pop'] > 0])
    unassigned = valid.copy()
    labels = {i: None for i in gdf.index}

    dem_votes = lambda i: gdf.at[i, 'Democratic_votes']
    rep_votes = lambda i: gdf.at[i, 'Republican_votes']
    share = lambda votes: (lambda i: votes(i) / gdf.at[i, 'precinct_pop']
                           if gdf.at[i, 'precinct_pop'] > 0 else -float('inf'))

    if maximize.lower() == 'gop':
        seed_metric = share(rep_votes)
        frontier_metric = lambda i: (rep_votes(i) - dem_votes(i)) / gdf.at[i, 'precinct_pop']
    else:
        seed_metric = share(dem_votes)
        frontier_metric = lambda i: (dem_votes(i) - rep_votes(i)) / gdf.at[i, 'precinct_pop']

    for d in range(NUM_DISTRICTS):
        if not unassigned:
            break
        seed = max(unassigned, key=seed_metric)
        block = {seed}
        pop = gdf.at[seed, 'precinct_pop']
        unassigned.remove(seed)

        while pop < min_pop and unassigned:
            frontier = {nbr for n in block for nbr in G.neighbors(n) if nbr in unassigned}
            if not frontier:
                break
            best = max(frontier, key=frontier_metric)
            block.add(best)
            pop += gdf.at[best, 'precinct_pop']
            unassigned.remove(best)
            if pop > max_pop:
                break

        for i in block:
            labels[i] = d

    # check on unassigned districts, find closest/most fitting
    district_pops = Counter()
    for i, lab in labels.items():
        if lab is not None:
            district_pops[lab] += gdf.at[i, 'precinct_pop']

    for i in [i for i, lab in labels.items() if lab is None]:
        neighbors = [labels[n] for n in G.neighbors(i) if labels[n] is not None]
        if neighbors:
            choice = Counter(neighbors).most_common(1)[0][0]
        else:
            choice = min(district_pops, key=lambda d: district_pops[d])
        labels[i] = choice
        district_pops[choice] += gdf.at[i, 'precinct_pop']

    return labels

# used to enforce contiguity rules, ensure that there are no 
# districts that are not mostly connected to their district 
def fix_contiguity(labels, G):
    changed = True
    while changed:
        changed = False
        for d in range(NUM_DISTRICTS):
            members = [i for i, lab in labels.items() if lab == d]
            comps = list(nx.connected_components(G.subgraph(members)))
            if len(comps) <= 1:
                continue
            main = max(comps, key=len)
            for comp in comps:
                if comp is main:
                    continue
                for node in comp:
                    nbrs = [labels[n] for n in G.neighbors(node) if labels[n] != d]
                    if nbrs:
                        labels[node] = Counter(nbrs).most_common(1)[0][0]
                        changed = True
    return labels

def post_balance_pop_per_district(labels, gdf, G):
    total_pop = gdf['precinct_pop'].sum()
    target = total_pop / NUM_DISTRICTS
    min_pop = target * (1 - POP_TOLERANCE)
    max_pop = target * (1 + POP_TOLERANCE)
    labels = labels.copy()

    while True:
        pops = {d: 0 for d in range(NUM_DISTRICTS)}
        for i, lab in labels.items():
            pops[lab] += gdf.at[i, 'precinct_pop']
        out_of = [d for d, p in pops.items() if p < min_pop or p > max_pop]
        if not out_of:
            break

        adjusted = False
        for d in out_of:
            if pops[d] < min_pop:
                boundary = {nbr for i, lab in labels.items() if lab == d
                            for nbr in G.neighbors(i) if labels[nbr] != d}
                for neighboring_node in boundary:
                    d2 = labels[neighboring_node]
                    new_pop_d = pops[d] + gdf.at[neighboring_node, 'precinct_pop']
                    new_pop_d2 = pops[d2] - gdf.at[neighboring_node, 'precinct_pop']
                    if min_pop <= new_pop_d <= max_pop and min_pop <= new_pop_d2 <= max_pop:
                        comp = set(i for i, lab in labels.items() if lab == d2)
                        comp.remove(neighboring_node)
                        if nx.is_connected(G.subgraph(comp)):
                            labels[neighboring_node] = d
                            adjusted = True
                            break
                if adjusted:
                    break
            else:
                for i, lab in labels.items():
                    if lab != d:
                        continue
                    for neighboring_node in G.neighbors(i):
                        d2 = labels[neighboring_node]
                        if d2 == d:
                            continue
                        new_pop_d = pops[d] - gdf.at[i, 'precinct_pop']
                        new_pop_d2 = pops[d2] + gdf.at[i, 'precinct_pop']
                        if min_pop <= new_pop_d <= max_pop and min_pop <= new_pop_d2 <= max_pop:
                            comp = set(j for j, lab_j in labels.items() if lab_j == d)
                            comp.remove(i)
                            if nx.is_connected(G.subgraph(comp)):
                                labels[i] = d2
                                adjusted = True
                                break
                    if adjusted:
                        break
                if adjusted:
                    break
        if not adjusted:
            break

        labels = fix_contiguity(labels, G)

    return labels

def main():
    parser = argparse.ArgumentParser(description='Greedy districting with party maximization.')
    parser.add_argument('--maximize', choices=['dem', 'gop'], default='dem',
                        help='Which party to maximize')
    args = parser.parse_args()

    gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=3857)
    gdf = compute_votes(gdf)
    G = create_adjacency_graph(gdf) #gets adjacency

    labels = greedy_logic(gdf, G, maximize=args.maximize)
    labels = fix_contiguity(labels, G)
    labels = post_balance_pop_per_district(labels, gdf, G)
    labels = fix_contiguity(labels, G)

    gdf['district'] = gdf.index.map(labels)
    gdf.to_file('districted_precincts.shp')

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap('tab20', NUM_DISTRICTS)
    gdf.plot(
        column='district', cmap=cmap, categorical=True,
        legend=True, legend_kwds={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'},
        linewidth=0.2, edgecolor='white', ax=ax
    )
    districts = gdf.dissolve(by='district')
    districts.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    for d, row in districts.iterrows():
        c = row.geometry.centroid
        ax.text(c.x, c.y, str(int(d)), ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_title(f"Districting (Packing {args.maximize.upper()} Votes)")
    ax.axis('off')
    plt.tight_layout()

    # Save versioned output
    output_dir = 'greedy_packing_figures'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(__file__))[0]
    png_pattern = os.path.join(output_dir, f"{base_name}_v*.png")
    version = len(glob.glob(png_pattern)) + 1
    png_path = os.path.join(output_dir, f"{base_name}_v{version}.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to {png_path}")
    csv_path = os.path.join(output_dir, f"{base_name}_v{version}.csv")
    gdf[['district']].to_csv(csv_path, index=True)
    print(f"Saved precinct mapping to {csv_path}")

    # Seat summary (how many gop vs dem)
    district_votes = gdf.groupby('district')[['Democratic_votes', 'Republican_votes']].sum()
    district_votes['Winner'] = district_votes.apply(
        lambda r: 'Democratic' if r['Democratic_votes'] > r['Republican_votes'] else 'Republican',
        axis=1
    )
    district_votes['Margin %'] = (
        (district_votes['Democratic_votes'] - district_votes['Republican_votes']) /
        (district_votes['Democratic_votes'] + district_votes['Republican_votes']) * 100
    ).round(2)

    print("\nVote Summary by District:")
    print(district_votes[['Democratic_votes', 'Republican_votes', 'Winner', 'Margin %']])
    print("\nSeat Totals:")
    print("  Democratic seats:", (district_votes['Winner'] == 'Democratic').sum())
    print("  Republican seats:", (district_votes['Winner'] == 'Republican').sum())

    plt.show()

if __name__ == '__main__':
    main()
