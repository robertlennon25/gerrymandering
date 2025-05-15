
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


#uses precinct-level voter data and census-block-level data 
# to predict voter outcomes based on population 

# this is not integrated with any of the other data, but could be used to 
# predict vote totals when they are not avialable 

# Load precinct shapefile (with congressional vote columns)
precincts = gpd.read_file("/precinct_level_data/il_2022_gen_prec(2)/il_2022_gen_cong_prec/il_2022_gen_cong_prec.shp")

# Load census block-level voting-age population data (P3)
census = gpd.read_file("/il_pl2020_b (1)/il_pl2020_p3_b.shp")
census = census.to_crs(precincts.crs)

# Spatial join - assignw each block to a precinct
joined = gpd.sjoin(census, precincts, how='left', predicate='within')

# Aggregate census block data to precinct level
agg = joined.groupby("Precinct").agg({
    "P0030001": "sum",  # total VAP (voting age population)
    "P0030003": "sum",  # White VAP
    "P0030004": "sum",  # Black VAP
    "P0030006": "sum",  # Asian VAP
}).reset_index()

# Calculate % by race
agg["pct_white_vap"] = agg["P0030003"] / agg["P0030001"]
agg["pct_black_vap"] = agg["P0030004"] / agg["P0030001"]
agg["pct_asian_vap"] = agg["P0030006"] / agg["P0030001"]

# Define the columns related to each district's candidates' votes
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

# Function to total congressional votes per precinct
def calculate_total_votes(row, candidate_info):
    dem_total = 0
    rep_total = 0
    for race in candidate_info.values():
        dem_total += row[race['Democratic']]
        rep_total += row[race['Republican']]
    return pd.Series({'Total_Democratic_Votes': dem_total, 'Total_Republican_Votes': rep_total})

# Apply to precincts
vote_totals = precincts.apply(calculate_total_votes, axis=1, candidate_info=candidate_info)
precincts = precincts.join(vote_totals)

# Drop the individual race columns
vote_columns_to_drop = [v for d in candidate_info.values() for v in d.values()]
precincts = precincts.drop(columns=vote_columns_to_drop)

# Join census VAP data to precincts
df = pd.merge(precincts, agg, on="Precinct", how="left")
df = df.dropna()

# Prepare regression data
X = df[["pct_white_vap", "pct_black_vap", "pct_asian_vap"]]
y_dem = df["Total_Democratic_Votes"]
y_rep = df["Total_Republican_Votes"]

# compute linear regression coefficients
import numpy as np

# Add intercept term 
X_mat = np.column_stack((np.ones(X.shape[0]), X.values))
y_dem_vec = y_dem.values
y_rep_vec = y_rep.values

# Beta
beta_dem = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_dem_vec
beta_rep = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_rep_vec

# Predictions
y_dem_pred = X_mat @ beta_dem
y_rep_pred = X_mat @ beta_rep

# Compute R^2 
r2_dem = 1 - np.sum((y_dem_vec - y_dem_pred) ** 2) / np.sum((y_dem_vec - np.mean(y_dem_vec)) ** 2)
r2_rep = 1 - np.sum((y_rep_vec - y_rep_pred) ** 2) / np.sum((y_rep_vec - np.mean(y_rep_vec)) ** 2)

print("Manual R² (Democratic votes):", r2_dem)
print("Manual R² (Republican votes):", r2_rep)

# Optional: plot predicted vs actual
import matplotlib.pyplot as plt
plt.scatter(y_dem_vec, y_dem_pred, alpha=0.5)
plt.xlabel("Actual Democratic Votes")
plt.ylabel("Predicted Democratic Votes")
plt.title("Democratic Vote Prediction (Linear Regression)")
plt.grid(True)
plt.show()
