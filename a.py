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

    # datasets["VestaCraters"] = pd.read_csv("data/vesta_craters.csv")
    # datasets["VestaCraters"]["Diameter"] *= 1e3
    datasets["VestaCraters"] = geopandas.read_file("data/vesta_crater_area_JF.shp")
    datasets["VestaCraters"]["A_m2"] = datasets["VestaCraters"].area

    datasets["MercuryCraters"] = pd.read_csv("data/mercury_craters.csv")
    datasets["MercuryCraters"]["Diameter"] *= 1e3

    return datasets

datasets = load_datasets()

labels = dict(WASLID="Earth (WA State DNR)", Crosta2018Mars="Mars (Crosta et al. 2018)", Brunetti2015Moon="Moon (Brunetti et al. 2015)", Brunetti2015Mercury="Mercury (Brunetti et al. 2015)", JFVesta="Vesta", VestaCraters="Vesta", MercuryCraters="Mercury (USGS)")

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
    plt.show()
    # plt.savefig("prob_area.pdf")
    plt.close()


def area_plot_zoomed_in():
    fig, ax = plt.subplots(layout="tight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Landslide Area [$m^2$]")
    plt.ylabel("Probability Density [$m^{-2}$]")
    for i,(key,dataset) in enumerate(datasets.items()):
        if key not in ['Brunetti2015Moon', 'Brunetti2015Mercury', 'JFVesta']:
            continue
        A = np.log(dataset["A_m2"].to_numpy())
        A = A[np.logical_not(np.isnan(A) | np.isinf(A))]
        x = np.linspace(A.min(), A.max())
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(A[:, None])
        probs = np.exp(kde.score_samples(x[:, None]))
        print(key, f"rollover = {np.exp(A)[(probs/np.exp(x)).argmax()]:0.2E}")
        plt.plot(np.exp(x), probs/np.exp(x), label=f"{labels[key]}, n={len(dataset)}", color=f"C{i}")
    plt.legend()
    plt.savefig("prob_area_zoom.pdf")
    plt.close()


def crater_area_plot():
    fig, ax = plt.subplots(layout="tight")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Crater Area [$m^{2}$]")
    plt.ylabel("Probability Density [$m^{-2}$]")
    colors = dict(Brunetti2015Mercury="C3", Brunetti2015Moon="C2", VestaCraters="C4")
    for key in ["Brunetti2015Moon", "Brunetti2015Mercury"]:
        m = datasets[key]
        D = m["Diameter"].unique()
        A = np.log(np.pi*(D/2)**2)
        x = np.linspace(A.min(), A.max())
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(A[:, None])
        probs = np.exp(kde.score_samples(x[:, None]))
        print(key, f"rollover = {np.exp(A)[(probs/np.exp(x)).argmax()]:0.2E}")
        ax.plot(np.exp(x), probs/np.exp(x), label=f"{labels[key]}, n={len(D)}", color=colors[key])
    # Do Vesta craters.
    key = "VestaCraters"
    A = np.log(datasets["VestaCraters"]["A_m2"].to_numpy())
    x = np.linspace(A.min(), A.max())
    kde = KernelDensity(kernel="gaussian", bandwidth="scott")
    kde.fit(A[:, None])
    probs = np.exp(kde.score_samples(x[:, None]))
    ax.plot(np.exp(x), probs/np.exp(x), label=f"{labels[key]}, n={len(A)}", color=colors[key])
    plt.legend()
    # plt.show()
    plt.savefig("crater_area.pdf")
    plt.close()
