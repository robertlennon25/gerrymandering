
import geopandas as gpd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained

# Configuration
SHAPEFILE_PATH = '/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp'
FIG_DIR = 'figures_sectionv5'
os.makedirs(FIG_DIR, exist_ok=True)
N_DISTRICTS = 17

# Load and preprocess shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)
gdf = gdf.to_crs(epsg=3857)

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

# Helper function: since KMeansConstrained doesn't use sample_weight, we weight the
# sample manually by normalizing population on smaller scale, then 
# extending samples depending upon the population of the precinct
def scale_population_weight(coords, population, scale=100):
    """
    Expands coordinate array by population weight.

    Parameters:
    - coords: np.array of shape (n, 2), one row per precinct
    - population: list or array of population per precinct
    - scale: normalization factor for how many repetitions per person

    Returns:
    - expanded_coords: np.array of repeated coordinates proportional to population
    - index_map: list mapping back to original precinct index
    """
    # Normalize to reasonable integers
    norm_weights = np.clip((population / population.max()) * scale, 1, scale).astype(int)
    expanded_coords = []
    index_map = []

    for i, (pt, w) in enumerate(zip(coords, norm_weights)):
        expanded_coords.extend([pt] * w)
        index_map.extend([i] * w)

    return np.array(expanded_coords), index_map

# preprocessing
gdf = compute_votes(gdf)
coords = np.vstack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).T
precinct_pop = gdf['precinct_pop'].values
state_total = precinct_pop.sum()
ideal_pop = state_total / N_DISTRICTS
min_pop = 0.95 * ideal_pop
max_pop = 1.05 * ideal_pop


expanded_coords, index_map = scale_population_weight(coords, precinct_pop, scale=100)

# Apply constrained KMeans, from the installed package
kmeans = KMeansConstrained(
    n_clusters=N_DISTRICTS,
    size_min=int(len(expanded_coords) / N_DISTRICTS * 0.95),
    size_max=int(len(expanded_coords) / N_DISTRICTS * 1.05),
    random_state=42
)
kmeans.fit(expanded_coords)
expanded_labels = kmeans.labels_

# reduce to label per original precinct via mode
labels = pd.Series(expanded_labels).groupby(index_map).agg(lambda x: x.mode().iloc[0]).values
gdf['district'] = labels + 1  # index districts 0-17

# save CSV
gdf[['Precinct', 'district']].to_csv('figures_sectionv5/kmeans_constrained_districts.csv', index=False)

# Plot and save the output
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(column='district', cmap='tab20', linewidth=0.1, edgecolor='gray', ax=ax)
plt.title('Constrained K-Means Districts (Population Expanded)')
plt.axis('off')

# save path (optional)
v = 1
plot_path = os.path.join(FIG_DIR, f"kmeans_constrained_map_v{v}.png")
while os.path.exists(plot_path):
    v += 1
    plot_path = os.path.join(FIG_DIR, f"kmeans_constrained_map_v{v}.png")

fig.savefig(plot_path, dpi=150)
print(f"Saved constrained KMeans map to {plot_path}")

#writes to figures_sectionv5

