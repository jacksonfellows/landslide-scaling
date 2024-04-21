import geopandas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


def load_datasets():
    datasets = dict()

    # WA state landslide inventory
    datasets["WASLID"] = geopandas.read_file("./data/ger_portal_landslide_database/WGS_Landslides.gdb/", layer="landslide_deposit")
    datasets["WASLID"]["A_m2"] = datasets["WASLID"].area*0.0929 # Convert to m^2.
    datasets["WASLID"]["V_m3"] = datasets["WASLID"]["VOLUME_FT3"]*0.02832 # Convert to m^3.

    # Crosta et al. 2018 Mars landslide inventory
    datasets["Crosta2018Mars"] = geopandas.read_file("./data/ess2173-sup-0003-data_set_si-s01")
    datasets["Crosta2018Mars"]["A_m2"] = datasets["Crosta2018Mars"]["Area"]

    # Brunetti et al. 2015 Moon & Mercury
    datasets["Brunetti2015Moon"] = pd.read_csv("./data/Brunetti2015Table1.csv")
    datasets["Brunetti2015Moon"]["A_m2"] = datasets["Brunetti2015Moon"]["Landslide area"]
    datasets["Brunetti2015Mercury"] = pd.read_csv("./data/Brunetti2015Table2.csv")
    datasets["Brunetti2015Mercury"]["A_m2"] = datasets["Brunetti2015Mercury"]["Landslide area"]

    return datasets

datasets = load_datasets()

def area_plot():
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Landslide Area [$m^2$]")
    plt.ylabel("Probability Density [$m^{-2}$]")
    for i,(key,dataset) in enumerate(datasets.items()):
        A = np.log(dataset["A_m2"].to_numpy())
        A = A[np.logical_not(np.isnan(A) | np.isinf(A))]
        x = np.linspace(A.min(), A.max())
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(A[:, None])
        probs = np.exp(kde.score_samples(x[:, None]))
        plt.plot(np.exp(x), probs/np.exp(x), label=f"{key} (n={len(dataset)})")
    plt.legend()
    plt.savefig("prob_area.pdf")

def area_vol_plot():
    A = gdf["A_m2"]
    V = gdf["V_m3"]

    I = (A != 0) & np.logical_not(np.isnan(V))
    log_A = np.log10(A[I])
    log_V = np.log10(V[I])
    gamma, log10a = np.polyfit(log_A, log_V, 1)
    # print(f"{gamma=:0.2f}, {log10a=:0.2f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Landslide Area [m^2]")
    plt.ylabel("Landslide Volume [m^3]")
    plt.scatter(A, V, facecolors="none", edgecolors="b", alpha=0.1, label="WA Landslide Inventory")

    # Draw power-law fit.
    A_ = np.array([10**1, 10**8])
    plt.plot(A_, 10**log10a*A_**gamma, "r--", label=f"Power Law Fit; $V = 10^{{{log10a:0.2f}}}A^{{{gamma:0.2f}}}$")

    # Label some specific landslides.
    oso_i = np.nonzero(gdf["LS_NAME"].str.contains("Oso") == True)[0][0]
    plt.annotate("Oso", (A[oso_i], V[oso_i]), xytext=(10,10), textcoords="offset points", arrowprops=dict(color="k", width=1, headwidth=4, headlength=4))

    plt.legend()

    plt.show()
