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

    datasets["JFVesta"] = geopandas.read_file("data/vesta_slide_area_JF.shp")
    datasets["JFVesta"]["A_m2"] = datasets["JFVesta"].area

    return datasets

datasets = load_datasets()

labels = dict(WASLID="Earth (WA State DNR)", Crosta2018Mars="Mars (Crosta et al. 2018)", Brunetti2015Moon="Moon (Brunetti et al. 2015)", Brunetti2015Mercury="Mercury (Brunetti et al. 2015)", JFVesta="Vesta")

def area_plot():
    fig, ax = plt.subplots(layout="tight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
        plt.plot(np.exp(x), probs/np.exp(x), label=f"{labels[key]}, n={len(dataset)}")
    plt.legend()
    plt.savefig("prob_area.pdf")
    plt.close()
