import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Path to the .shp file (ensure .shx, .dbf, etc. are in same folder)
shapefile_path = "/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp"

# Load shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Define vote columns
candidate_info = {
    '1st Congress': {'Democratic': 'GCON01DJAC', 'Republican': 'GCON01RCAR'},
    '2nd Congress': {'Democratic': 'GCON02DKEL', 'Republican': 'GCON02RLYN'},
    '3rd Congress': {'Democratic': 'GCON03DRAM', 'Republican': 'GCON03RBUR'},
    '4th Congress': {'Democratic': 'GCON04DGAR', 'Republican': 'GCON04RFAL'},
    '5th Congress': {'Democratic': 'GCON05DQUI', 'Republican': 'GCON05RHAN'},
    '6th Congress': {'Democratic': 'GCON06DCAS', 'Republican': 'GCON06RPEK'},
    '7th Congress': {'Democratic': 'GCON07DDAV', 'Republican': 'GCON07OWRI'},
    '8th Congress': {'Democratic': 'GCON08DKRI', 'Republican': 'GCON08RDAR'},
    '9th Congress': {'Democratic': 'GCON09DSCH', 'Republican': 'GCON09RRIC'},
    '10th Congress': {'Democratic': 'GCON10DSCH', 'Republican': 'GCON10RSEV'},
    '11th Congress': {'Democratic': 'GCON11DFOS', 'Republican': 'GCON11RLAU'},
    '12th Congress': {'Democratic': 'GCON12DMAR', 'Republican': 'GCON12RBOS'},
    '13th Congress': {'Democratic': 'GCON13DBUD', 'Republican': 'GCON13RDEE'},
    '14th Congress': {'Democratic': 'GCON14DUND', 'Republican': 'GCON14RGRY'},
    '15th Congress': {'Democratic': 'GCON15DLAN', 'Republican': 'GCON15RMIL'},
    '16th Congress': {'Democratic': 'GCON16DHAD', 'Republican': 'GCON16RLAH'},
    '17th Congress': {'Democratic': 'GCON17DSOR', 'Republican': 'GCON17RKIN'}
}

# Compute vote ratio
def compute_vote_ratio(row):
    dem, rep = 0, 0
    for cand in candidate_info.values():
        dem += row.get(cand['Democratic'], 0) or 0
        rep += row.get(cand['Republican'], 0) or 0
    total = dem + rep
    return rep / total if total > 0 else 0.5

gdf["vote_ratio"] = gdf.apply(compute_vote_ratio, axis=1)

# plot
fig, ax = plt.subplots(figsize=(10, 12))
cmap = plt.cm.bwr
norm = mcolors.Normalize(vmin=0, vmax=1)

gdf.plot(column="vote_ratio", cmap=cmap, norm=norm, linewidth=0.1, edgecolor="gray", ax=ax)
ax.set_title("Vote Ratio by Precinct (2022 Illinois)", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.show()
