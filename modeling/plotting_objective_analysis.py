import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from scipy import stats

def process_dictionary(dicts_or_stat, interaction_info, joint_entropy_dict, conditional_entropies):
    # plot average entropy vs compression ratio

    stat_dict_df = pd.DataFrame(dicts_or_stat)
    ds_name = stat_dict_df["Name"][0]
    combined_ent = stat_dict_df["entropy"]
    combined_entk2 = stat_dict_df["k2 entropy"]
    CR_original = stat_dict_df["original cr"]
    CR_reordered = stat_dict_df["reordered cr"]
    CR_decomp = stat_dict_df["decomposed cr"]

    comp_Tool = stat_dict_df["tool"][0]
    average_ent, std_ent, std_ent2 = [], [], []
    for v in combined_ent:
        average_ent.append(np.mean(v))
        std_ent.append(np.std(v))
    for v in combined_entk2:
        average_ent.append(np.mean(v))
        std_ent2.append(np.std(v))
    # create a dataframe
    df = pd.DataFrame({
        "STD Entropy": std_ent,
        "STD Entropy K2": std_ent2,
        "Compression Ratio": CR_decomp / CR_original,
    })
    # plot the data
    #sns.scatterplot(data=df, x="STD Entropy", y="Compression Ratio")
    # create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plot the first subplot
    sns.scatterplot(data=df, x="STD Entropy", y="Compression Ratio", ax=ax1)
    ax1.set_title(f"{ds_name} : {comp_Tool}")
    ax1.set_xlabel("STD Entropy")
    ax1.set_ylabel("Decomposed Compression Ratio")
    # add a regression line
    sns.regplot(data=df, x="STD Entropy", y="Compression Ratio", scatter=False, color='red', ax=ax1)
    # show the r2
    r2 = stats.linregress(df["STD Entropy"], df["Compression Ratio"]).rvalue ** 2
    print(f"R2 for STD Entropy vs Compression Ratio: {r2}")

    # plot the second subplot
    sns.scatterplot(data=df, x="STD Entropy K2", y="Compression Ratio", ax=ax2)
    ax2.set_title(f"{ds_name} : {comp_Tool}")
    ax2.set_xlabel("STD Entropy K2")
    ax2.set_ylabel("Decomposed Compression Ratio")
    sns.regplot(data=df, x="STD Entropy K2", y="Compression Ratio", scatter=False, color='blue', ax=ax2)
    # show the r2
    r2 = stats.linregress(df["STD Entropy K2"], df["Compression Ratio"]).rvalue ** 2
    print(f"R2 for STD Entropy K2 vs Compression Ratio: {r2}")
    plt.show()


df = pd.read_csv("all_clustering.csv")
process_dictionary(df, None, None, None)