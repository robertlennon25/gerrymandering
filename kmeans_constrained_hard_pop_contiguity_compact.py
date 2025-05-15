import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
# from postprocess_isolated_precincts import postprocess_isolated_precincts

# Configuration
SHAPEFILE_PATH = '/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp'
N_DISTRICTS = 17
MAX_ITER = 100
OUT_DIR = 'kmeans_constrained_hard_pop'
os.makedirs(OUT_DIR, exist_ok=True)

# Load shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)
gdf = gdf.to_crs(epsg=3857)

# Compute precinct population from votes
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
        gdf['Democratic_votes'] += gdf[dv]
        gdf['Republican_votes'] += gdf[rv]
    gdf['precinct_pop'] = gdf['Democratic_votes'] + gdf['Republican_votes']
    return gdf

gdf = compute_votes(gdf)

# Build adjacency list
gdf['geometry'] = gdf.geometry.buffer(0)  # fix geometry errors
sindex = gdf.sindex
adj = {i: [] for i in range(len(gdf))}
for i, geom in enumerate(gdf.geometry):
    for j in sindex.intersection(geom.bounds):
        if i < j and geom.touches(gdf.geometry[j]):
            adj[i].append(j)
            adj[j].append(i)

# Get coordinates and population
coords = np.vstack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).T
population = gdf['precinct_pop'].values

# Population constraints
state_total = population.sum()
ideal_pop = state_total / N_DISTRICTS
max_pop = 1.05 * ideal_pop
alpha = 10000  # population penalty weight
contiguity_bonus = 2000  # reward for placing next to same-cluster precinct

# Initialize centers randomly
np.random.seed(42)
initial_index = np.random.choice(len(coords), N_DISTRICTS, replace=False)
centers = coords[initial_index].copy()

# Initialize labels and cluster pops
labels = np.full(len(coords), -1, dtype=int)
cluster_pop = [0] * N_DISTRICTS

# KMeans with soft population constraint and contiguity bonus
for it in range(MAX_ITER):
    changed = False
    for i in range(len(coords)):
        best_score = float('inf')
        best_c = -1
        for c in range(N_DISTRICTS):
            if cluster_pop[c] + population[i] > max_pop:
                continue  # Hard cap on population

            dist = np.linalg.norm(coords[i] - centers[c])
            penalty = alpha * ((cluster_pop[c] + population[i] - ideal_pop) / ideal_pop) ** 2
            score = dist + penalty

            # Add contiguity bonus
            if any(labels[n] == c for n in adj[i]):
                score -= contiguity_bonus

            if score < best_score:
                best_score = score
                best_c = c

        if best_c != -1 and labels[i] != best_c:
            if labels[i] != -1:
                cluster_pop[labels[i]] -= population[i]
            cluster_pop[best_c] += population[i]
            labels[i] = best_c
            changed = True

    # Recompute centers
    for c in range(N_DISTRICTS):
        members = coords[labels == c]
        if len(members) > 0:
            centers[c] = np.mean(members, axis=0)

    if not changed:
        print(f"Converged at iteration {it}")
        break



# Assign labels to GeoDataFrame
gdf['district'] = labels + 1

# After clustering is complete
# gdf = postprocess_isolated_precincts(gdf, population_col='precinct_pop')


# Compute compactness metrics for each district
districts = gdf.dissolve(by='district')
districts['area'] = districts.geometry.area
districts['perimeter'] = districts.geometry.length
districts['compactness_sqspace'] = districts['area'] / districts['area'].max()
districts['compactness_polsby'] = (4 * np.pi * districts['area']) / (districts['perimeter'] ** 2)

# Save CSV with district sizes
district_sizes = pd.DataFrame({'district': range(1, N_DISTRICTS+1), 'population': cluster_pop})

# Add compactness metrics
districts = districts.reset_index()
district_sizes = district_sizes.merge(
    districts[['district', 'compactness_sqspace', 'compactness_polsby']],
    on='district',
    how='left'
)
csv_version = 1
while os.path.exists(os.path.join(OUT_DIR, f'district_sizes_{csv_version}.csv')):
    csv_version += 1
csv_path = os.path.join(OUT_DIR, f'district_sizes_{csv_version}.csv')
district_sizes.to_csv(csv_path, index=False)
print(f"Saved population CSV to {csv_path}")

# Save map figure
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(column='district', cmap='tab20', linewidth=0.1, edgecolor='gray', ax=ax)
plt.title("Soft-Constrained KMeans Districts (with Contiguity Bonus)")
plt.axis('off')

fig_version = 1
while os.path.exists(os.path.join(OUT_DIR, f'figure_{fig_version}.png')):
    fig_version += 1
fig_path = os.path.join(OUT_DIR, f'figure_{fig_version}.png')
fig.savefig(fig_path, dpi=150)
print(f"Saved district map to {fig_path}")
