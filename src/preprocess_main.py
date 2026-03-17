import argparse
import os
import re
import time

from Bio import SeqIO
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from util.my import time_elapsed

SLKG_KG_path = "../data_raw/SLKG2/raw_kg.tsv"
protein_sequence_path = "../data_raw/uniprot/uniprotkb_organism_id_9606_AND_reviewed_2024_10_26.fasta"
PAN_CN_KG = "TOTAL"


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ct",
        nargs="+",
        type=str,
        default=["BRCA", "CESC", "COAD", "KIRC", "LAML", "LUAD", "OV", "SKCM"],
        # default=["pan"],
        help="one or more cancer types",
    )
    parser.add_argument(
        "--cn_kg",
        nargs="+",
        type=str,
        default=["breast cancer", "cervical cancer", "colon cancer", "kidney cancer", "hematologic cancer", "lung cancer", "ovarian cancer", "skin cancer"],
        # default=["TOTAL"],
        help="node name of `Disease` in KG. each value is corresponding to a ct parameter. `EMPTY` for no corresponding cancer, `TOTAL` for pan cancer.",
    )
    parser.add_argument(
        "--omics_types",
        nargs="+",
        type=str,
        default=["cna", "exp", "mut", "myl"],
        # default=["cna", "exp", "mut"],
        help="name of omics category used",
    )

    return parser.parse_args()


###################################################################################### preprocess_tcga below


def test_and_modify_omics_data(df, name):
    nan_cells_count = df.isnull().sum(axis=1).sum()
    print(name, "nan_cells:", nan_cells_count)
    if nan_cells_count > 0:
        df = df.fillna(0)  # Set NaN positions to 0

    # Convert non-numeric values to NaN
    numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    no_numeric_count = numeric_df.isna().sum(axis=1).sum()
    if no_numeric_count > 0:
        numeric_df = numeric_df.fillna(0)  # Set NaN positions to 0
        df = pd.concat([df.iloc[:, 0], numeric_df.iloc[:, 1:]], axis=1)
    print(name, "no_numeric_cells:", no_numeric_count)

    print("success to modify", name)
    return df


def transform_mut_format(df_mut):
    """
    Original mutation data is a list of gene and mutation names, need to convert to matrix format similar to other 3 omics
    """

    all_samples = df_mut["Tumor_Sample_Barcode"].drop_duplicates()
    new_df = pd.DataFrame(index=df_mut["Hugo_Symbol"].unique())
    new_df = new_df.reindex(columns=all_samples, fill_value=0)
    print(f"transforming mutation data format, {len(df_mut)} rows...")
    for _, row in tqdm(df_mut.iterrows(), total=len(df_mut)):
        new_df.loc[row["Hugo_Symbol"], row["Tumor_Sample_Barcode"]] += 1  # Here mutation count is used as value
    print("finished.")
    return new_df


def preprocess_tcga(cancer_types, omics_types):
    for cancer_type in cancer_types:
        print(f"======================================== preprocess tcga about {cancer_type} ========================================")
        # Get gene names
        gene_idx_to_name = torch.load(f"../data/{cancer_type}/gene.pt", weights_only=False)
        gene_names = list(gene_idx_to_name.values())
        gene_names_index = pd.Index(gene_names).astype(str)

        # Read data
        df_omics_dict = {}
        for omics_type in omics_types:
            omics_path = f"../data_raw/TCGA/{cancer_type}/{omics_type}.txt"
            print(f"reading {omics_type} data...")
            df_omics = pd.read_csv(omics_path, sep="\t", low_memory=False)
            df_omics_dict[omics_type] = df_omics

        for omics_type, df_omics in df_omics_dict.items():
            if omics_type == "mut":
                df_omics = transform_mut_format(df_omics_dict["mut"])
            else:
                if "Entrez_Gene_Id" in df_omics.columns:
                    df_omics = df_omics.drop(columns=["Entrez_Gene_Id"])
                df_omics = df_omics.drop_duplicates("Hugo_Symbol")
                df_omics.set_index("Hugo_Symbol", inplace=True)

            df_omics_dict[omics_type] = df_omics

        # Get all sample column names from all files
        all_columns = []
        for omics_type, df_omics in df_omics_dict.items():
            all_columns.extend(df_omics.columns.tolist())
        all_samples = pd.Index(all_columns).unique()
        all_samples = all_samples.astype(str)
        all_samples = np.sort(all_samples)

        for omics_type, df_omics in df_omics_dict.items():
            print(f"rebuilding {omics_type}...")
            df_omics_reindex = df_omics.reindex(index=gene_names_index, columns=all_samples, fill_value=0)
            df_omics_reindex = df_omics_reindex.reset_index().rename(columns={"index": "Hugo_Symbol"})
            df_omics_reindex = test_and_modify_omics_data(df_omics_reindex, omics_type)

            df_omics_dict[omics_type] = df_omics_reindex

        if "exp" in df_omics_dict:
            df_omics_dict["exp"].iloc[:, 1:] = df_omics_dict["exp"].iloc[:, 1:].clip(lower=-10.0)

        # Save extended DataFrame to new file
        directory = f"../data/{cancer_type}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        for omics_type, df_omics in df_omics_dict.items():
            df_omics = df_omics.iloc[:, 1:]
            np_omics = df_omics.to_numpy()

            # Log-transform mutation data and scale to [0,1]
            if omics_type == "mut":
                max_value = np_omics.max()
                if max_value > 0:
                    np_omics = np.log1p(np_omics) / np.log1p(max_value)
                else:
                    np_omics = np.zeros_like(np_omics)

            cur_omics_save_path = f"../data/{cancer_type}/{omics_type}.npy"
            np.save(cur_omics_save_path, np_omics)
            print(f"success to save {omics_type} data about {cancer_type} to {cur_omics_save_path}")


###################################################################################### preprocess_tcga above
###################################################################################### preprocess_kg below


def generate_specific_edges_data(cancer_type, cancer_name_in_kg, df_slkg, entity_to_index):
    print(f"======================================== preprocess KG edges about {cancer_type} ========================================")

    # Drop edges related to Disease not of current cancer
    if cancer_name_in_kg.strip() == PAN_CN_KG:
        print('pan cancer will use all "Disease" nodes in KG')
    else:
        df_slkg_cancer = df_slkg[(df_slkg["x_type"] == "Disease") | (df_slkg["y_type"] == "Disease")]
        df_slkg_cancer_drop_x = df_slkg_cancer[(df_slkg_cancer["x_name"] != cancer_name_in_kg) & (df_slkg_cancer["x_type"] == "Disease")]
        df_slkg_cancer_drop_y = df_slkg_cancer[(df_slkg_cancer["y_name"] != cancer_name_in_kg) & (df_slkg_cancer["y_type"] == "Disease")]
        df_slkg_cancer_drop = pd.concat([df_slkg_cancer_drop_x, df_slkg_cancer_drop_y])
        print(f'the count of dropping "Disease" related edges whose name is not "{cancer_name_in_kg}": {len(df_slkg_cancer_drop)}')
        df_slkg = df_slkg.drop(df_slkg_cancer_drop.index)

    # Remove direct connection edges of unrelated genes. Operation below avoids deleting "A (model gene) - B (not model gene)" causing A to be unrepresentable
    gene_idx_to_name = torch.load(f"../data/{cancer_type}/gene.pt", weights_only=False)
    slgenes = list(gene_idx_to_name.values())
    df_slkg_drop_gene_x = df_slkg[(df_slkg["x_type"] == "Gene") & (~df_slkg["x_name"].isin(slgenes)) & (df_slkg["y_type"] != "Gene")]
    df_slkg_drop_gene_y = df_slkg[(df_slkg["y_type"] == "Gene") & (~df_slkg["y_name"].isin(slgenes)) & (df_slkg["x_type"] != "Gene")]
    df_slkg_drop_gene_xy = df_slkg[(df_slkg["x_type"] == "Gene") & (df_slkg["y_type"] == "Gene") & (~df_slkg["x_name"].isin(slgenes)) & (~df_slkg["y_name"].isin(slgenes))]
    df_slkg_drop_gene = pd.concat([df_slkg_drop_gene_x, df_slkg_drop_gene_y, df_slkg_drop_gene_xy])
    print(f"kg edges count of dropping unrelated genes: {len(df_slkg_drop_gene)}")
    df_slkg = df_slkg.drop(df_slkg_drop_gene.index)

    df_slkg["relation_reversed"] = df_slkg["relation"] + "_reversed"
    graph_edges_data = {}

    # Batch process forward edges
    for (x_type, y_type), group in df_slkg.groupby(["x_type", "y_type"]):
        print(f"handling KG edge: {x_type} -> {y_type}")
        relations = group["relation"].unique()
        for relation in relations:
            relation_edges = group[group["relation"] == relation]
            x_id = [entity_to_index[key] for key in relation_edges["x_id"].values]
            y_id = [entity_to_index[key] for key in relation_edges["y_id"].values]
            edge_index_batch = torch.tensor([x_id, y_id], dtype=torch.long)
            unique_edges = torch.unique(edge_index_batch, dim=1)
            graph_edges_data[(x_type, relation, y_type)] = unique_edges

    # Batch process reverse edges
    for (y_type, x_type), group in df_slkg.groupby(["y_type", "x_type"]):
        print(f"handling KG edge reversed: {y_type} -> {x_type}")
        reversed_relations = group["relation_reversed"].unique()
        for reversed_relation in reversed_relations:
            if reversed_relation in ["REGULATES_GrG"]:  # If stored as undirected (bidirectional) in graph, no need to add reversed edge
                continue

            reversed_edges = group[group["relation_reversed"] == reversed_relation]
            y_id = [entity_to_index[key] for key in reversed_edges["y_id"].values]
            x_id = [entity_to_index[key] for key in reversed_edges["x_id"].values]
            edge_index_reversed_batch = torch.tensor([y_id, x_id], dtype=torch.long)
            unique_edges = torch.unique(edge_index_reversed_batch, dim=1)
            graph_edges_data[(y_type, reversed_relation, x_type)] = unique_edges

    return graph_edges_data


def modify_gene_id_with_duplicate_gene_names(df_slkg):
    """
    Some genes with same name have multiple IDs, unify to one
    """

    df_slkg_gene_x = df_slkg[df_slkg["x_type"] == "Gene"]
    df_slkg_gene_x = df_slkg_gene_x[["x_id", "x_name"]].rename(columns={"x_id": "id", "x_name": "name"})
    df_slkg_gene_y = df_slkg[df_slkg["y_type"] == "Gene"]
    df_slkg_gene_y = df_slkg_gene_y[["y_id", "y_name"]].rename(columns={"y_id": "id", "y_name": "name"})
    df_slkg_gene = pd.concat([df_slkg_gene_x, df_slkg_gene_y], ignore_index=True).drop_duplicates()

    df_duplicate = df_slkg_gene.groupby("name").filter(lambda x: x["id"].nunique() > 1)
    df_duplicate["id"] = pd.to_numeric(df_duplicate["id"], errors="coerce")
    df_duplicate = df_duplicate.sort_values(by=["name", "id"], ascending=[True, False])
    print("duplicate_gene_name:\n", df_duplicate)

    name_id_dict = dict(zip(df_duplicate["name"], df_duplicate["id"]))
    for key, value in name_id_dict.items():
        df_slkg.loc[(df_slkg["x_type"] == "Gene") & (df_slkg["x_name"] == key), "x_id"] = value
        df_slkg.loc[(df_slkg["y_type"] == "Gene") & (df_slkg["y_name"] == key), "y_id"] = value

    print("success to modify duplicate gene id.")
    return df_slkg


def generate_generic_nodes_data():
    print(f"======================================== preprocess KG generic nodes ========================================")
    print("reading SLKG 2.0 data, please wait...")
    df_slkg_generic = pd.read_csv(SLKG_KG_path, sep="\t", low_memory=False)
    df_slkg_drop = df_slkg_generic[df_slkg_generic["relation"].isin(["SL_GsG", "NONSL_GnsG", "SR_GsrG"])]  # Drop SL related edges
    print(f"total count of KG edges: {len(df_slkg_generic)}. drop [SL_GsG,NONSL_GnsG,SR_GsrG] relations: {len(df_slkg_drop)}")
    df_slkg_generic = df_slkg_generic.drop(df_slkg_drop.index)

    df_slkg_generic = modify_gene_id_with_duplicate_gene_names(df_slkg_generic)

    print("generating KG nodes...")
    df_nodes = pd.concat(
        [
            df_slkg_generic.get(["x_id", "x_name", "x_type"]).rename(columns={"x_id": "id", "x_name": "name", "x_type": "type"}),
            df_slkg_generic.get(["y_id", "y_name", "y_type"]).rename(columns={"y_id": "id", "y_name": "name", "y_type": "type"}),
        ],
        ignore_index=True,
    )
    df_nodes = df_nodes.drop_duplicates().reset_index().drop("index", axis=1)
    print(f"generic KG nodes count:", len(df_nodes))

    # Get all entity names, create dict and bidirectional reference
    all_entities = set(df_nodes["id"].unique())
    entity_to_index = {entity: idx + 2 for idx, entity in enumerate(sorted(all_entities))}
    entity_to_index["<PAD>"] = 0
    entity_to_index["<MASK>"] = 1
    index_to_entity = {idx: entity for entity, idx in entity_to_index.items()}

    # Map KG index to system gene name
    df_entity_gene = df_nodes[df_nodes["type"] == "Gene"]
    entity_to_gene_name = dict(zip(df_entity_gene["id"], df_entity_gene["name"]))  # entity_id -> name
    entity_idx_to_gene_name = {}
    for k, v in entity_to_gene_name.items():
        entity_idx_to_gene_name[entity_to_index[k]] = v

    gene_name_to_entity_idx = {value: key for key, value in entity_idx_to_gene_name.items()}

    # Build heterogeneous network nodes by node type
    graph_nodes_data = {}
    for node_type in df_nodes["type"].unique():
        print(f"handling KG node: {node_type}")
        node_df = df_nodes[df_nodes["type"] == node_type]

        ids = node_df["id"].values
        node_ids = [entity_to_index[key] for key in ids]
        node_ids_tensor = torch.tensor(node_ids, dtype=torch.long)
        node_ids_unique = torch.unique(node_ids_tensor, return_inverse=False)
        graph_nodes_data[node_type] = node_ids_unique

    return df_slkg_generic, entity_to_index, index_to_entity, entity_idx_to_gene_name, gene_name_to_entity_idx, graph_nodes_data


def preprocess_kg(cancer_types, cancer_names_in_kg):
    df_slkg_generic, entity_to_index, index_to_entity, entity_idx_to_gene_name, gene_name_to_entity_idx, graph_nodes_data = generate_generic_nodes_data()

    for index, cancer_type in enumerate(cancer_types):
        df_slkg = df_slkg_generic.copy()
        graph_edges_data = generate_specific_edges_data(cancer_type, cancer_names_in_kg[index], df_slkg, entity_to_index)

        kg_graph = HeteroData()
        for key, value in graph_nodes_data.items():
            kg_graph[key].node_id = value
        for key, value in graph_edges_data.items():
            kg_graph[key].edge_index = value

        data_dict = {
            "entity_to_index": entity_to_index,
            "index_to_entity": index_to_entity,
            "entity_idx_to_gene_name": entity_idx_to_gene_name,
            "gene_name_to_entity_idx": gene_name_to_entity_idx,
            "kg_graph": kg_graph,
        }
        kg_data_save_path = f"../data/{cancer_type}/kg.pt"
        torch.save(data_dict, kg_data_save_path)
        print(f"success to save kg related data to:", kg_data_save_path)


###################################################################################### preprocess_kg above
###################################################################################### preprocess_sequence below


def preprocess_sequence(cancer_types):
    """
    有些基因有多个蛋白质序列，要选择最长的 Some genes have multiple protein sequences, choose longest
    """

    # Protein sequence
    records = list(SeqIO.parse(protein_sequence_path, "fasta"))
    gene_sequences = []
    for record in records:
        match = re.search(r"GN=(.*?) PE", record.description)
        if match:
            gn_value = match.group(1)
            sequence = "".join(record.seq)
            gene_sequences.append((gn_value, sequence))
    df_total = pd.DataFrame(gene_sequences, columns=["Hugo_Symbol", "Sequence"])

    # Some genes have multiple protein sequences due to alternative splicing, choose longest
    df_total = df_total.loc[df_total.groupby("Hugo_Symbol")["Sequence"].apply(lambda x: x.str.len().idxmax())]

    for cancer_type in cancer_types:
        print(f"======================================== preprocess sequence about {cancer_type} ========================================")
        gene_idx_to_name = torch.load(f"../data/{cancer_type}/gene.pt", weights_only=False)
        slgenes = list(gene_idx_to_name.values())
        df_specific = df_total[df_total["Hugo_Symbol"].isin(slgenes)]

        # Sort and save to file
        df_specific = df_specific.sort_values(by=["Hugo_Symbol"])
        sequence_save_path = f"../data/{cancer_type}/sequence.csv"
        df_specific.to_csv(sequence_save_path, index=False)
        print(f"success to save sequence data about {cancer_type} to:", sequence_save_path)


###################################################################################### preprocess_sequence above
###################################################################################### preprocess_SL below


def margin_genes(ELISL_SL_pairs_df, gene_sequence_list, cancer_type, cancer_name_in_kg, omics_types):
    """
    return:
        - intersect_genes: gene Genes present in sequence data, KG, and TCGA
        - ELISL_SL_pairs_df: SL dataframe filtered by intersect_genes
    """

    # Knowledge Graph
    print("reading SLKG 2.0 data, please wait...")
    df_slkg = pd.read_csv(SLKG_KG_path, sep="\t", low_memory=False)
    df_slkg_drop = df_slkg[df_slkg["relation"].isin(["SL_GsG", "NONSL_GnsG", "SR_GsrG"])]  # Drop SL related edges
    print(f"total count of KG edges: {len(df_slkg)}. drop [SL_GsG,NONSL_GnsG,SR_GsrG] edges: {len(df_slkg_drop)}")
    df_slkg = df_slkg.drop(df_slkg_drop.index)

    # Drop edges related to Disease not of current cancer
    if cancer_name_in_kg.strip() == PAN_CN_KG:
        print('pan cancer will use all "Disease" nodes in KG')
    else:
        df_slkg_cancer = df_slkg[(df_slkg["x_type"] == "Disease") | (df_slkg["y_type"] == "Disease")]
        df_slkg_cancer_drop_x = df_slkg_cancer[(df_slkg_cancer["x_name"] != cancer_name_in_kg) & (df_slkg_cancer["x_type"] == "Disease")]
        df_slkg_cancer_drop_y = df_slkg_cancer[(df_slkg_cancer["y_name"] != cancer_name_in_kg) & (df_slkg_cancer["y_type"] == "Disease")]
        df_slkg_cancer_drop = pd.concat([df_slkg_cancer_drop_x, df_slkg_cancer_drop_y])
        print(f'the count of dropping "Disease" related edges whose name is not "{cancer_name_in_kg}": {len(df_slkg_cancer_drop)}')
        df_slkg = df_slkg.drop(df_slkg_cancer_drop.index)

    # Get genes present in graph
    df_slkg_x_name = df_slkg[df_slkg["x_type"] == "Gene"]["x_name"]
    df_slkg_y_name = df_slkg[df_slkg["y_type"] == "Gene"]["y_name"]
    df_slkg_genes = pd.concat([df_slkg_x_name, df_slkg_y_name], ignore_index=True).drop_duplicates()

    # SL genes
    slgene1 = ELISL_SL_pairs_df["gene1"]
    slgene2 = ELISL_SL_pairs_df["gene2"]
    slgenes = pd.concat([slgene1, slgene2], ignore_index=True).drop_duplicates()

    # TCGA genes
    df_omics_array = []
    for omics_type in omics_types:
        omics_path = f"../data_raw/TCGA/{cancer_type}/{omics_type}.txt"
        print(f"reading {omics_type} data...")
        df_omics = pd.read_csv(omics_path, sep="\t", low_memory=False)
        df_omics = df_omics["Hugo_Symbol"]
        df_omics_array.append(df_omics)
    df_TCGA_merged_genes = pd.concat(df_omics_array, ignore_index=True).drop_duplicates()

    # Get intersection of data sources
    intersect_genes = df_slkg_genes[df_slkg_genes.isin(gene_sequence_list)]
    intersect_genes = intersect_genes[intersect_genes.isin(df_TCGA_merged_genes)]
    print("genes count used in this cancer:", len(intersect_genes))

    # Exclude SL genes that are not present in the data sources
    remove_SL_genes = slgenes[~slgenes.isin(intersect_genes)]
    print("SL genes count not in our multi resources:", len(remove_SL_genes))
    ELISL_SL_pairs_df = ELISL_SL_pairs_df[(~ELISL_SL_pairs_df["gene1"].isin(remove_SL_genes)) & (~ELISL_SL_pairs_df["gene2"].isin(remove_SL_genes))]
    positive_samples = ELISL_SL_pairs_df[ELISL_SL_pairs_df["class"] == 1]
    negative_samples = ELISL_SL_pairs_df[ELISL_SL_pairs_df["class"] == 0]
    print("current positive samples:", len(positive_samples), "negative samples:", len(negative_samples), "total samples:", len(ELISL_SL_pairs_df))
    return intersect_genes, ELISL_SL_pairs_df


def preprocess_SL(cancer_types, cancer_names_in_kg, omics_types):
    # Protein sequence
    records = list(SeqIO.parse(protein_sequence_path, "fasta"))
    gene_list = []  # Gene list of protein sequence info
    for record in records:
        match = re.search(r"GN=(.*?) PE", record.description)
        if match:  # Some sequences have no gene name
            gn_value = match.group(1)
            gene_list.append(gn_value)

    for index, cancer_type in enumerate(cancer_types):
        print(f"======================================== preprocess SL about {cancer_type} ========================================")
        ELISL_SL_pairs_df = pd.read_csv(f"../data_raw/SL/{cancer_type}.csv")
        print("original SL count:", len(ELISL_SL_pairs_df))

        # Align Genes from multi-source data
        intersect_genes, ELISL_SL_pairs_df = margin_genes(ELISL_SL_pairs_df, gene_list, cancer_type, cancer_names_in_kg[index], omics_types)

        # gene
        intersect_genes = intersect_genes.sort_values()
        intersect_genes = intersect_genes.reset_index().drop("index", axis=1)
        gene_idx_to_name = intersect_genes[0].to_dict()
        genes_name_path = f"../data/{cancer_type}/gene.pt"
        directory = os.path.dirname(genes_name_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(gene_idx_to_name, genes_name_path)
        print(f"success to save all genes idx to name about {cancer_type} to:", genes_name_path)

        # SL
        ELISL_SL_pairs_df = ELISL_SL_pairs_df.sort_values(by=["gene1", "gene2"])
        gene_name_to_idx = {value: key for key, value in gene_idx_to_name.items()}
        ELISL_SL_pairs_df["gene1"] = ELISL_SL_pairs_df["gene1"].map(gene_name_to_idx)
        ELISL_SL_pairs_df["gene2"] = ELISL_SL_pairs_df["gene2"].map(gene_name_to_idx)
        sl_array = ELISL_SL_pairs_df.to_numpy()
        full_SL_pairs_save_path = f"../data/{cancer_type}/sl.npy"
        np.save(full_SL_pairs_save_path, sl_array)
        print(f"success to save full SL data about {cancer_type} to:", full_SL_pairs_save_path)


###################################################################################### preprocess_SL above


def main():
    """
    Preprocess data
    """

    args = init_argparse()
    print("parameters:", args)
    cancer_types = args.ct
    cancer_names_in_kg = args.cn_kg
    omics_types = args.omics_types

    start_time = time.time()
    print("*********************preprocess SL*********************")
    preprocess_SL(cancer_types, cancer_names_in_kg, omics_types)
    print("*********************preprocess sequence*********************")
    preprocess_sequence(cancer_types)
    print("*********************preprocess KG*********************")
    preprocess_kg(cancer_types, cancer_names_in_kg)
    print("*********************preprocess tcga*********************")
    preprocess_tcga(cancer_types, omics_types)
    end_time = time.time()
    print()
    print(time_elapsed(start_time, end_time, "All preprocess finished! Time used: "))


if __name__ == "__main__":
    main()
