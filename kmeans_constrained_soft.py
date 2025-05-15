import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter

# Configuration
SHAPEFILE_PATH = '/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp'
N_DISTRICTS = 17
BASE_RANDOM_SEED = 43
MAX_ITER = 100
FIG_DIR = 'figures_sectionv3'
BASE_FILENAME = 'constrained_kmeans'
OUTPUT_CSV = 'constrained_kmeans_districts.csv'
CONTIGUITY_BONUS = 10       # reward for contiguity
MIN_POP_BONUS = 1000        # reward for reaching lower pop threshold
LOWER_POP_FACTOR = 0.9      # fraction of ideal to reward
RESTARTS = 5                # number of random restarts

# Load and preprocess shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)
gdf = gdf.to_crs(epsg=3857)  # project to a planar CRS

# Sum votes per precinct
def compute_votes(gdf):
    mapping = {
        '1st':('GCON01DJAC','GCON01RCAR'),'2nd':('GCON02DKEL','GCON02RLYN'),
        '3rd':('GCON03DRAM','GCON03RBUR'),'4th':('GCON04DGAR','GCON04RFAL'),
        '5th':('GCON05DQUI','GCON05RHAN'),'6th':('GCON06DCAS','GCON06RPEK'),
        '7th':('GCON07DDAV','GCON07OWRI'),'8th':('GCON08DKRI','GCON08RDAR'),
        '9th':('GCON09DSCH','GCON09RRIC'),'10th':('GCON10DSCH','GCON10RSEV'),
        '11th':('GCON11DFOS','GCON11RLAU'),'12th':('GCON12DMAR','GCON12RBOS'),
        '13th':('GCON13DBUD','GCON13RDEE'),'14th':('GCON14DUND','GCON14RGRY'),
        '15th':('GCON15DLAN','GCON15RMIL'),'16th':('GCON16DHAD','GCON16RLAH'),
        '17th':('GCON17DSOR','GCON17RKIN')
    }
    gdf['Democratic_votes'] = 0
    gdf['Republican_votes'] = 0
    for dv, rv in mapping.values():
        gdf['Democratic_votes'] += gdf[dv]
        gdf['Republican_votes'] += gdf[rv]
    gdf['precinct_pop'] = gdf['Democratic_votes'] + gdf['Republican_votes']
    return gdf

gdf = compute_votes(gdf)
precinct_pop = gdf['precinct_pop'].values
state_total = precinct_pop.sum()
ideal_pop = state_total / N_DISTRICTS
pop_threshold = 1.05 * ideal_pop
min_pop_thresh = LOWER_POP_FACTOR * ideal_pop
print(f"Ideal pop: {ideal_pop:.0f}, max threshold: {pop_threshold:.0f}, reward threshold: {min_pop_thresh:.0f}")

# Build adjacency list for contiguity
gdf['geometry'] = gdf.geometry.buffer(0)
sindex = gdf.sindex
adj = {i: [] for i in range(len(gdf))}
for i, geom in enumerate(gdf.geometry):
    for j in sindex.intersection(geom.bounds):
        if i < j and geom.touches(gdf.geometry[j]):
            adj[i].append(j)
            adj[j].append(i)

# Compute centroids array
coords = np.vstack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).T

# Helper to generate versioned file paths
def next_versioned_path(base, ext):
    os.makedirs(FIG_DIR, exist_ok=True)
    v = 1
    path = os.path.join(FIG_DIR, f"{base}_v{v}.{ext}")
    while os.path.exists(path):
        v += 1
        path = os.path.join(FIG_DIR, f"{base}_v{v}.{ext}")
    return path

# Single-seed constrained k-means
def run_seed(seed):
    random.seed(seed)
    seed_idx = random.sample(range(len(coords)), N_DISTRICTS)
    centroids = coords[seed_idx].copy()
    labels = np.full(len(coords), -1, dtype=int)
    cluster_pop = np.zeros(N_DISTRICTS)
    for k, i in enumerate(seed_idx):
        labels[i] = k
        cluster_pop[k] = precinct_pop[i]

    def assign_labels():
        new_labels = np.full(len(coords), -1, dtype=int)
        # lock seeds
        for k, i in enumerate(seed_idx):
            new_labels[i] = k
        # assign others
        for i in range(len(coords)):
            if new_labels[i] != -1:
                continue
            costs = []
            for k in range(N_DISTRICTS):
                # hard cap: cluster removed if at threshold
                if cluster_pop[k] >= pop_threshold:
                    costs.append(np.inf)
                    continue
                new_pop = cluster_pop[k] + precinct_pop[i]
                if new_pop > pop_threshold:
                    costs.append(np.inf)
                    continue
                # distance cost
                d = np.sum((coords[i] - centroids[k])**2)
                cost = d
                # contiguity bonus
                if any(new_labels[n] == k for n in adj[i]):
                    cost -= CONTIGUITY_BONUS
                # reward for reaching lower threshold
                if new_pop >= min_pop_thresh:
                    cost -= MIN_POP_BONUS
                costs.append(cost)
            best_k = int(np.argmin(costs))
            new_labels[i] = best_k
        return new_labels

    # iterate
    for _ in range(MAX_ITER):
        labels = assign_labels()
        # update centroids and pops
        for k in range(N_DISTRICTS):
            members = np.where(labels == k)[0]
            if len(members) > 0:
                centroids[k] = coords[members].mean(axis=0)
                cluster_pop[k] = precinct_pop[members].sum()
    # compute pop deviation
    deviation = pd.Series(cluster_pop).max() - pd.Series(cluster_pop).min()
    return labels, deviation

# Run multiple restarts and pick best
best_dev = np.inf
best_labels = None
for i in range(RESTARTS):
    seed = BASE_RANDOM_SEED + i
    labels, dev = run_seed(seed)
    print(f"Restart {i+1} (seed {seed}): pop deviation = {dev:.0f}")
    if dev < best_dev:
        best_dev, best_labels = dev, labels.copy()
print(f"Selected best pop deviation: {best_dev:.0f}")

# Remove enclaves: ensure each district is one connected component
# Build global adjacency graph
G = nx.Graph()
G.add_nodes_from(range(len(coords)))
for i, nbrs in adj.items():
    for j in nbrs:
        if i < j:
            G.add_edge(i, j)
labels = best_labels.copy()
changed = True
while changed:
    changed = False
    for k in range(N_DISTRICTS):
        nodes_k = [i for i, lab in enumerate(labels) if lab == k]
        sub = G.subgraph(nodes_k)
        comps = list(nx.connected_components(sub))
        if len(comps) <= 1:
            continue
        # keep largest component, reassign islands
        comps.sort(key=len, reverse=True)
        for island in comps[1:]:
            for precinct in island:
                neigh_labels = [labels[n] for n in G.neighbors(precinct) if labels[n] != k]
                if not neigh_labels:
                    continue
                new_k = Counter(neigh_labels).most_common(1)[0][0]
                labels[precinct] = new_k
                changed = True

# Save district assignments
gdf['district'] = labels + 1
gdf[['Precinct','district']].to_csv(OUTPUT_CSV, index=False)
print(f"Saved assignments to {OUTPUT_CSV}")

# Compute final stats
stats = gdf.groupby('district').agg(
    Dem=('Democratic_votes','sum'),
    Rep=('Republican_votes','sum')
)
stats['Total'] = stats['Dem'] + stats['Rep']
stats['MarginPct'] = (stats['Dem'] - stats['Rep']).abs() / stats['Total'] * 100
stats['Winner'] = np.where(stats['Dem'] > stats['Rep'], 'D', 'R')
dems = (stats['Winner'] == 'D').sum()
reps = N_DISTRICTS - dems

# Save summary text
summary_path = next_versioned_path(f"{BASE_FILENAME}_summary","txt")
with open(summary_path,'w') as f:
    for d, row in stats.iterrows():
        f.write(f"Dist {d}: {row['Winner']} by {row['MarginPct']:.1f}% ({int(row['Total'])} votes)\n")
    f.write(f"Total D seats: {dems}, R seats: {reps}\n")
print(f"Summary saved to {summary_path}")

# Plot final map
fig, ax = plt.subplots(1,1,figsize=(12,12))
gdf.plot(column='district', cmap='tab20', linewidth=0.1, edgecolor='gray', ax=ax)
for d, geom in gdf.dissolve('district').geometry.items():
    x, y = geom.centroid.x, geom.centroid.y
    w = stats.loc[d, 'Winner']
    pct = stats.loc[d, 'MarginPct']
    ax.text(x, y, f"{w}{pct:.0f}%", ha='center', va='center', fontsize=8)
plt.figtext(0.5, 0.02, f"D seats: {dems}   R seats: {reps}", ha='center')
plt.axis('off')
map_path = next_versioned_path(f"{BASE_FILENAME}_map","png")
fig.savefig(map_path, dpi=150)
print(f"Map saved to {map_path}")


# --- Version 2: Population-Constrained KMeans ---

def population_constrained_kmeans(X, population, k, max_iter=100, alpha=5.0):
    """
    K-means clustering with a soft population constraint.

    Parameters:
    - X: np.array of shape (n_samples, n_features), e.g., coordinates
    - population: list or array of length n_samples with population per point
    - k: number of clusters
    - max_iter: number of iterations
    - alpha: population penalty weight

    Returns:
    - labels: array of cluster assignments
    """
    n = len(X)
    labels = np.full(n, -1)
    centers = X[np.random.choice(n, k, replace=False)]
    ideal_pop = sum(population) / k
    cluster_pop = [0] * k

    for it in range(max_iter):
        changed = False

        for i in range(n):
            best_dist = float("inf")
            best_c = None

            for c in range(k):
                dist = np.linalg.norm(X[i] - centers[c])
                penalty = alpha * abs(cluster_pop[c] + population[i] - ideal_pop)
                score = dist + penalty

                if score < best_dist:
                    best_dist = score
                    best_c = c

            if best_c is not None and labels[i] != best_c:
                changed = True
                if labels[i] != -1:
                    cluster_pop[labels[i]] -= population[i]
                labels[i] = best_c
                cluster_pop[best_c] += population[i]

        if not changed:
            break

        for c in range(k):
            pts = X[labels == c]
            if len(pts) > 0:
                centers[c] = np.mean(pts, axis=0)

    return labels
