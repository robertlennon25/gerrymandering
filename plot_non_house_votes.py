import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_il2020_by_county(shapefile_path="il_2020/il_2020.shp", out_dir="figures_section_v1"):
    print("Loading shapefile...")
    gdf = gpd.read_file(shapefile_path)

    print("Aggregating by county...")
    vote_columns = {
        "pres_dem": "G20PREDBID",
        "pres_gop": "G20PRERTRU",
        "sen_dem": "G20USSDDUR",
        "sen_gop": "G20USSRCUR"
    }

    for alias, col in vote_columns.items():
        gdf[alias] = gdf[col].fillna(0)

    grouped = gdf.groupby("COUNTYFP20").agg({
        "pres_dem": "sum",
        "pres_gop": "sum",
        "sen_dem": "sum",
        "sen_gop": "sum"
    }).reset_index()

    grouped["dem_total"] = grouped["pres_dem"] + grouped["sen_dem"]
    grouped["gop_total"] = grouped["pres_gop"] + grouped["sen_gop"]
    grouped["total_votes"] = grouped["dem_total"] + grouped["gop_total"]
    grouped["dem_pct"] = grouped["dem_total"] / grouped["total_votes"]
    grouped["gop_pct"] = grouped["gop_total"] / grouped["total_votes"]


    county_geom = gdf.dissolve(by="COUNTYFP20", as_index=False)
    merged = county_geom.merge(grouped, on="COUNTYFP20")

    fig, ax = plt.subplots(figsize=(12, 12))
    merged.plot(
        column="gop_pct",
        cmap="bwr",
        legend=True,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        vmin=0,
        vmax=1
    )

    ax.set_title("2020 Estimated Partisan Lean by County (President + Senate)", fontsize=14)
    ax.axis("off")

    # Label each county
    for _, row in merged.iterrows():
        fips = row["COUNTYFP20"]
        pct = row["dem_pct"]
        #label is not 100*pct-50, but instead 50*pct-25, to avoid doubly-counting party-line voters
        label = f"D+{int(pct*50 - 25)}%" if pct >= 0.5 else f"R+{int((1 - pct)*50 - 25)}%"
        centroid = row.geometry.centroid

        ax.text(
            centroid.x, centroid.y, label,
            fontsize=7, weight='bold', color='white',
            ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2')
        )

    plt.tight_layout()

    # Save (optional)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith("county_redblue_v") and f.endswith(".png")]
    existing_versions = [int(f.split("v")[-1].split(".")[0]) for f in existing if f.split("v")[-1].split(".")[0].isdigit()]
    next_version = max(existing_versions, default=0) + 1
    output_path = os.path.join(out_dir, f"county_redblue_v{next_version}.png")

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"✅ Saved to: {output_path}")


if __name__ == "__main__":
    plot_il2020_by_county()
