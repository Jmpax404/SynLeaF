import os
import time

import pandas as pd
from tqdm import tqdm

from util.my import time_elapsed


def transform_slkg_2_to_3_format():
    """
    SLKG 2.0 has no duplicate edges, `REGULATES_GrG` relation is bidirectional, `Disease` node is cancer-specific
    """

    # Read raw data
    slkg2_raw_data_path = "../data_raw/SLKG2/sldb_complete.csv"
    print(f"reading SLKG2.0: {slkg2_raw_data_path}")
    df_kg_raw = pd.read_csv(slkg2_raw_data_path, low_memory=False)

    # Get node attributes
    df_kg_entity = df_kg_raw[df_kg_raw["_id"].notna()]
    print("handling entities fields...")
    # _id is node ID, convert float to int. _labels is relation, remove leading colon
    entity_inf_dict = {int(row["_id"]): [row["name"], row["_labels"][1:]] for _, row in df_kg_entity.iterrows()}

    # Get edge information
    df_kg_edge = df_kg_raw[df_kg_raw["_start"].notna()]
    x_id_array = []
    x_type_array = []
    x_name_array = []
    y_id_array = []
    y_type_array = []
    y_name_array = []
    relation_array = []
    print(f"transforming edges data, {len(df_kg_edge)} rows...")
    for _, row in tqdm(df_kg_edge.iterrows(), total=len(df_kg_edge)):
        x_id = int(row["_start"])
        y_id = int(row["_end"])
        relation = row["_type"]

        x_fields = entity_inf_dict[x_id]
        x_name = x_fields[0]
        x_type = x_fields[1]

        y_fields = entity_inf_dict[y_id]
        y_name = y_fields[0]
        y_type = y_fields[1]

        x_id_array.append(x_id)
        x_type_array.append(x_type)
        x_name_array.append(x_name)
        y_id_array.append(y_id)
        y_type_array.append(y_type)
        y_name_array.append(y_name)
        relation_array.append(relation)
    print("finished.")

    # Save to file
    new_kg_data = {
        "x_id": x_id_array,
        "x_type": x_type_array,
        "x_name": x_name_array,
        "y_id": y_id_array,
        "y_type": y_type_array,
        "y_name": y_name_array,
        "relation": relation_array,
    }

    df_new_kg = pd.DataFrame(new_kg_data)
    df_new_kg.to_csv("../data_raw/SLKG2/raw_kg.tsv", index=False, sep="\t")
    print(f"success to save KG with new format to", "../data_raw/SLKG2/raw_kg.tsv")


def preprocess_ELISL():
    # Merge ELISL training set and test set. No bidirectional ones
    ELISL_train_pairs_df = pd.read_csv("../data_raw/ELISL/train_pairs.csv")
    ELISL_test_pairs_df = pd.read_csv("../data_raw/ELISL/test_pairs.csv")
    ELISL_SL_pairs_df = pd.concat([ELISL_train_pairs_df, ELISL_test_pairs_df], ignore_index=True).drop_duplicates()
    print("ELISL_SL_pairs original count:", len(ELISL_SL_pairs_df))

    # Change SL relation of A-A from 1 to 0
    ELISL_SL_pairs_df.loc[ELISL_SL_pairs_df["gene1"] == ELISL_SL_pairs_df["gene2"], "class"] = 0

    # Separate samples
    df_grouped = ELISL_SL_pairs_df.groupby(["cancer"])
    for name, df_group in df_grouped:
        cancer_type = name[0]

        directory = "../data_raw/SL"
        if not os.path.exists(directory):
            os.makedirs(directory)

        df_group = df_group.get(["gene1", "gene2", "class"])
        df_group.to_csv(f"{directory}/{cancer_type}.csv", index=False)
        print(f"success to save SL from ELISL about {cancer_type}.")


def generate_pan_sl():
    # Positive samples, did not save bidirectional SL simultaneously, no self-loops
    df_sl_pos = pd.read_csv("../data_raw/SLKG2/Human_SL.csv", low_memory=False)
    df_sl_pos = df_sl_pos[["n1.name", "n2.name"]].rename(columns={"n1.name": "gene1", "n2.name": "gene2"})
    print("raw SL(positive) samples:", len(df_sl_pos))

    df_non_sl = pd.read_csv("../data_raw/SLKG2/Human_nonSL.csv", low_memory=False)
    df_non_sl = df_non_sl[["n1.name", "n2.name"]].rename(columns={"n1.name": "gene1", "n2.name": "gene2"})
    print("raw non SL samples:", len(df_non_sl))

    df_sr = pd.read_csv("../data_raw/SLKG2/Human_SR.csv", low_memory=False)
    df_sr = df_sr[["n1.name", "n2.name"]].rename(columns={"n1.name": "gene1", "n2.name": "gene2"})
    print("raw SR samples:", len(df_sr))

    df_sl_neg = pd.concat([df_non_sl, df_sr], ignore_index=True)
    df_sl_neg = df_sl_neg.drop_duplicates()
    print("raw negative samples:", len(df_sl_neg))

    # Neither nonSL nor SR saved bidirectional edges, but combining them creates duplicates/bidirectionals
    df_sl_neg.loc[:, "gene_pair"] = df_sl_neg.apply(lambda row: tuple(sorted([row["gene1"], row["gene2"]])), axis=1)
    df_sl_neg = df_sl_neg.drop_duplicates(subset="gene_pair", keep="first")
    df_sl_neg = df_sl_neg.drop(columns=["gene_pair"])
    print("no bidirectional negative samples:", len(df_sl_neg))

    # When gene pair is both positive and negative, treat as negative
    df_sl_pos.loc[:, "gene_pair"] = df_sl_pos.apply(lambda row: tuple(sorted([row["gene1"], row["gene2"]])), axis=1)
    df_sl_neg.loc[:, "gene_pair"] = df_sl_neg.apply(lambda row: tuple(sorted([row["gene1"], row["gene2"]])), axis=1)
    common_pairs = set(df_sl_pos["gene_pair"]).intersection(set(df_sl_neg["gene_pair"]))
    print(f"some samples are both pos and neg: {len(common_pairs)}, which will be considered as negative samples.")
    df_sl_pos = df_sl_pos[~df_sl_pos["gene_pair"].isin(common_pairs)].copy()
    df_sl_pos = df_sl_pos.drop(columns=["gene_pair"])
    df_sl_neg = df_sl_neg.drop(columns=["gene_pair"])

    # Set type
    df_sl_pos["class"] = 1
    df_sl_neg["class"] = 0
    df_sl = pd.concat([df_sl_pos, df_sl_neg], ignore_index=True)
    print(f"before further processing, positive SL: {len(df_sl_pos)}, negative SL: {len(df_sl_neg)}.")

    directory = "../data_raw/SL"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_sl.to_csv(f"{directory}/pan.csv", index=False)
    print("success to save pan SL from SLDB 2.0 to", f"{directory}/pan.csv")


def main():
    """
    Pre-processing of pre-processing
    """

    start_time = time.time()
    print("*********************generate pan SL*********************")
    generate_pan_sl()
    print("*********************preprocess ELISL*********************")
    preprocess_ELISL()
    print("*********************transform KG format*********************")
    transform_slkg_2_to_3_format()
    end_time = time.time()
    print()
    print(time_elapsed(start_time, end_time, "All preprocess finished! Time used: "))


if __name__ == "__main__":
    main()
