import numpy as np
import torch
from torch.utils.data import Dataset

from util.my import rank_print


class MultiModalDataset(Dataset):
    def __init__(self, accelerator, gene_path, omics_path_dict, kg_graph_path):
        """
        Initialize Dataset object, read multi-modal data
        :param accelerator: Accelerator object
        :param gene_path: File path mapping gene index to name
        :param omics_path_dict: Dictionary of omics data, key is omics name, value is file path of omics data
        :param kg_graph_path: Path to knowledge graph data
        """

        ###################################################################################### Process gene-related data
        rank_print(accelerator, f"reading gene data: {gene_path}")
        gene_idx_to_name = torch.load(gene_path, weights_only=False)
        self.gene_idx_to_name = gene_idx_to_name
        self.gene_name_to_idx = {gene: idx for idx, gene in gene_idx_to_name.items()}

        ###################################################################################### Process omics-related data
        self.omics_count = len(omics_path_dict)  # Number of omics
        omics_data = {}
        for omics_name, csv_path in omics_path_dict.items():
            rank_print(accelerator, f"reading {omics_name} data: {csv_path}")
            np_omics = np.load(csv_path)
            omics_data[omics_name] = np_omics
        self.omics_data = omics_data  # Multi-omics data, storing numpy arrays for each omics

        ###################################################################################### Process kg-related data
        rank_print(accelerator, f"reading kg data: {kg_graph_path}")
        kg_data_dict = torch.load(kg_graph_path, weights_only=False)

        self.entity_to_index = kg_data_dict["entity_to_index"]
        self.index_to_entity = kg_data_dict["index_to_entity"]
        self.entity_idx_to_gene_name = kg_data_dict["entity_idx_to_gene_name"]
        self.gene_name_to_entity_idx = kg_data_dict["gene_name_to_entity_idx"]
        kg_graph = kg_data_dict["kg_graph"]  # Heterogeneous graph structure of the knowledge graph

        # Get all edges of the heterogeneous network
        all_edge_indices = []  # All edges
        all_edge_types = []  # Edge types
        type_index = 0
        for edge_type in kg_graph.edge_types:
            edge_index = kg_graph[edge_type].edge_index
            all_edge_indices.append(edge_index)

            edge_index_length = edge_index.size(1)  # Number of edges
            edge_type_tensor = torch.full((edge_index_length,), type_index, dtype=torch.long)  # Creating a tensor, assign the same type_index to all edges of this type
            all_edge_types.append(edge_type_tensor)
            type_index += 1  # Increment type_index for the next edge type to use a different index
        kg_graph.concatenated_edge_index = torch.cat(all_edge_indices, dim=1)  # Concatenate to get all edges, Tensor [2,N]
        kg_graph.concatenated_edge_type = torch.cat(all_edge_types)  # All edge types, Tensor [N]

        self.kg_graph = kg_graph

    def __len__(self):
        """
        Returns the size of the dataset
        """

        return len(self.gene_idx_to_name)

    def __getitem__(self, idx):
        """
        Get multi-modal data for the specified idx
        :param idx: Index in the dataset
        """

        # Extract multi-omics data from omics_data (iterate by omics name in dict)
        omics_data_row_tensor = []
        for omics_name, np_omics in self.omics_data.items():
            omic_value = np_omics[idx]
            omics_data_row_tensor.append(torch.Tensor(omic_value))

        # Get knowledge graph entity
        gene_name = self.gene_idx_to_name[idx]
        entity_idx = self.gene_name_to_entity_idx[gene_name]

        result = {
            "omics_data_list": omics_data_row_tensor,
            "gene_entity": torch.tensor(entity_idx, dtype=torch.long),
        }
        return result

    def get_omics_input_dim(self):
        """
        Get tissue omics case sample dimension
        """

        first_key = list(self.omics_data.keys())[0]
        return self.omics_data[first_key].shape[1]

    def get_kg_relations_count(self):
        return len(self.kg_graph.edge_types)

    def get_entity_vocab_size(self):
        """
        Get knowledge graph all entity vocabulary size, used for node embedding representation
        0 is <PAD>
        1 is <MASK>
        """

        return len(self.entity_to_index)

    def get_kg_graph(self):
        return self.kg_graph


class SLDataset(Dataset):
    def __init__(self, accelerator, sl_path, mm_dataset):
        """
        Initialize Dataset object, read SL data
        :param accelerator: Accelerator object
        :param sl_path: Synthetic lethality dataset path
        :param mm_dataset: Complete multi-modal dataset
        """

        rank_print(accelerator, f"reading SL data: {sl_path}")
        np_sl = np.load(sl_path)
        self.np_sl = np_sl
        self.mm_dataset = mm_dataset

    def __len__(self):
        """
        Returns the size of the dataset
        """

        return self.np_sl.shape[0]

    def __getitem__(self, idx):
        """
        Get relevant data for SL pairs. It will call __getitem__ of MultiModalDataset
        :param idx: Index in the SL dataset
        """

        sample = self.np_sl[idx]

        gene1_data = self.mm_dataset[sample[0]]
        gene2_data = self.mm_dataset[sample[1]]
        label = sample[2]

        gene1_data = {f"gene_1_{key}": value for key, value in gene1_data.items()}
        gene2_data = {f"gene_2_{key}": value for key, value in gene2_data.items()}

        # Merge
        result = gene1_data | gene2_data
        result["label"] = torch.tensor(label, dtype=torch.float)  # BCEWithLogitsLoss requires the target to be float type as well

        return result
