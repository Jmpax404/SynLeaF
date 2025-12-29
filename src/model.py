import numpy as np
import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import k_hop_subgraph
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RAdam
from util.lookahead import Lookahead

from util.my import rank_main_print


class VAE(nn.Module):
    """
    Self-modal Variational Autoencoder
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, dropout):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: list, multiple intermediate dimensions
            latent_dim: Output dimension
            dropout: Dropout rate for each layer
        """
        super().__init__()

        # Build multi-layer MLP
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            # When using SyncBatchNorm in distributed training, batch normalization statistics are synchronized across multiple GPUs to ensure all devices use the same mean and variance
            layers.append(nn.SyncBatchNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature_encoder = nn.Sequential(*layers)

        self.mu_predictor = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())
        self.log_var_predictor = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())

        # Prevent log_var from being too large, causing overflow when calculating KL divergence
        self.LOG_VAR_MIN = -10.0
        self.LOG_VAR_MAX = 10.0

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim], input features
        Returns:
            mu: mean
            log_var: log var
        """

        for layer in self.feature_encoder:
            x = layer(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)

        log_var = torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        return mu, log_var


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Networks
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers, graph_dropout):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        if n_layers == 1:  # Single layer
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        else:  # Multi-layer
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
            for i in range(n_layers - 2):
                self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

        self.graph_dropout = graph_dropout

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = F.dropout(x, p=self.graph_dropout, training=self.training)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index, edge_type)
        return x


class OmicsEncoder(nn.Module):
    """
    Omics Encoder
    """

    def __init__(self, params):
        """
        :param omics_count: Number of omics data
        :param omics_input_dim: Input dimension of omics data
        :param vae_hidden_dims: Intermediate dimensions of VAE
        :param vae_dropout: Dropout rate for each layer of VAE
        :param hid_dim: Dimension after Encoder
        """
        super().__init__()

        # Parse parameters
        omics_count = params["omics_count"]
        omics_input_dim = params["omics_input_dim"]
        vae_hidden_dims = params["vae_hidden_dims"]
        vae_dropout = params["vae_dropout"]
        hid_dim = params["hid_dim"]

        # Matrix-form VAE of size [omics_count, omics_count], containing self-VAE and cross-VAE
        self.omics_encoders = nn.ModuleList(nn.ModuleList([VAE(omics_input_dim, vae_hidden_dims, hid_dim, vae_dropout) for j in range(omics_count)]) for i in range(omics_count))
        self.hid_dim = hid_dim

        self.fc = nn.Linear(int(2 * hid_dim), hid_dim)

    def forward(self, params):
        """
        :param omics_data_list: [batch_size, omics_input_dim] List of omics data, each element shape is [batch_size, omics_input_dim]
        :param is_training: Whether in training phase, random noise should be sampled during training
        """
        omics_data_list = params["omics_data_list"]
        is_training = params["is_training"]

        current_device = omics_data_list[0].device
        omics_count = len(omics_data_list)

        # The first omics_count are encoded by self-VAE, the latter omics_count are encoded by cross-VAE
        vae_z = [None for _ in range(omics_count * 2)]

        self_kl_loss = torch.tensor(0.0).to(current_device)
        cross_kl_loss = torch.tensor(0.0).to(current_device)
        # Encode omics data and obtain tokens
        for i in range(omics_count):
            others_mu_array = []
            others_log_var_array = []
            for j in range(omics_count):
                if i == j:  # self-VAE
                    # [batch_size, omics_input_dim] -> [batch_size, (mu, log_var)]
                    mu, log_var = self.omics_encoders[i][j](omics_data_list[j])
                    self_kl_loss += OmicsEncoder.kl_loss(mu, log_var)

                    # [batch_size, hid_dim] -> [batch_size, 1, hid_dim]
                    if is_training:
                        vae_z[i] = OmicsEncoder.re_parameterize(mu, log_var).unsqueeze(1)
                    else:
                        vae_z[i] = mu.unsqueeze(1)  # To avoid randomness, non-training tasks can directly use the mean value as input

                else:  # cross-VAE
                    mu, log_var = self.omics_encoders[i][j](omics_data_list[j])
                    others_mu_array.append(mu)
                    others_log_var_array.append(log_var)

            # Calculate PoE
            poe_mu, poe_log_var = OmicsEncoder.product_of_experts(others_mu_array, others_log_var_array)
            cross_kl_loss += OmicsEncoder.kl_loss(poe_mu, poe_log_var)

            if is_training:
                vae_z[omics_count + i] = OmicsEncoder.re_parameterize(poe_mu, poe_log_var).unsqueeze(1)
            else:
                vae_z[omics_count + i] = poe_mu.unsqueeze(1)

        vae_z = torch.cat(vae_z, dim=1)  # Tensor, [batch_size, omics_count*2, vae_dim]

        self_tokens = vae_z[:, :omics_count, :]  # [batch_size, 4, hid_dim]
        cross_tokens = vae_z[:, omics_count:, :]  # [batch_size, 4, hid_dim]

        self_emb = self_tokens.mean(dim=1)  # [batch_size, hid_dim]
        cross_emb = cross_tokens.mean(dim=1)  # [batch_size, hid_dim]

        final_emb = torch.cat([self_emb, cross_emb], dim=-1)  # [batch_size, 2*hid_dim]
        final_emb = self.fc(final_emb)

        return final_emb, self_kl_loss, cross_kl_loss

    @staticmethod
    def product_of_experts(mu_set_, log_var_set_):
        """
        @From: https://github.com/FengAoWang/TMO-Net
        """

        tmp = 0
        for i in range(len(mu_set_)):
            tmp += torch.div(1, torch.exp(log_var_set_[i]))

        poe_var = torch.div(1.0, tmp)
        poe_log_var = torch.log(poe_var)

        tmp = 0.0
        for i in range(len(mu_set_)):
            tmp += torch.div(1.0, torch.exp(log_var_set_[i])) * mu_set_[i]
        poe_mu = poe_var * tmp
        return poe_mu, poe_log_var

    @staticmethod
    def re_parameterize(mean, log_var):
        """
        @From: https://github.com/FengAoWang/TMO-Net
        Returns: z
        """

        log_var = torch.exp(log_var / 2)  # log_var -> std
        epsilon = torch.randn_like(log_var)
        return epsilon * log_var + mean

    @staticmethod
    def kl_loss(mu, log_var, reduction="mean"):
        """
        KL divergence loss
        """

        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        if reduction == "mean":
            return kl.mean()  # Default average over B and d
        elif reduction == "sum":
            return kl.sum(dim=1).mean()


class KGEncoder(nn.Module):
    """
    Knowledge Graph Encoder
    """

    def __init__(self, params):
        """
        :param entity_vocab_size: Vocabulary size of all entities in KG; note 0 is <PAD>, 1 is <MASK>.
        :param hid_dim: GCN output dimension
        :param in_channels: Input dimension of KG entities
        :param hidden_channels: Hidden layer dimension of GNN (if multiple layers)
        :param gcn_layers: GCN layers
        :param graph_dropout: Graph dropout rate
        :param num_relations: Number of relations used in RGCN
        """
        super().__init__()

        # Parse parameters
        entity_vocab_size = params["entity_vocab_size"]
        hid_dim = params["hid_dim"]
        in_channels = params["in_channels"]
        hidden_channels = params["hidden_channels"]
        gcn_layers = params["gcn_layers"]
        graph_dropout = params["graph_dropout"]
        num_relations = params["num_relations"]

        # Knowledge graph entity encoder, size is all entities in the entire graph
        self.entity_encoder = nn.Embedding(entity_vocab_size, in_channels)
        # Some GNN network
        self.gnn = RGCN(in_channels, hidden_channels, hid_dim, num_relations, gcn_layers, graph_dropout)
        # GCN layers; sampling hops use the same
        self.gcn_layers = gcn_layers

    def forward(self, params):
        """
        :param gene_entity: Gene itself entity idx, [batch_size, 1]
        :param kg_graph: Heterogeneous graph structure of the knowledge graph
        """
        # Parameters
        gene_entity = params["gene_entity"]
        kg_graph = params["kg_graph"]

        current_device = gene_entity.device

        batch_unique_entities = torch.unique(gene_entity)  # [unique_gene_entities_count]

        # Get edges of the heterogeneous network
        concatenated_edge_index = kg_graph.concatenated_edge_index
        concatenated_edge_type = kg_graph.concatenated_edge_type

        # Sample surrounding neighbors. relabel_nodes=True makes node edge values be subset indices. The return values of these are all in CPU
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=batch_unique_entities,  # Index of center nodes, integer or list containing multiple node indices.
            num_hops=self.gcn_layers,
            edge_index=concatenated_edge_index,  # Edge index of the original graph, shape is [2, num_edges].
            relabel_nodes=True,
            flow="source_to_target",
            directed=False,
        )

        # Get values of concatenated_edge_type where edge_mask is True, forming a sub-tensor sequence of concatenated_edge_type
        sub_edge_type = concatenated_edge_type[edge_mask]

        # Put required values into GPU
        subset = subset.to(current_device)  # Node indices (or IDs) in the extracted subgraph.
        sub_edge_index = sub_edge_index.to(current_device)  # [2, num_edges_in_subgraph] Edge indices in the subgraph.
        sub_edge_type = sub_edge_type.to(current_device)  # Subgraph edge mask (boolean tensor). Boolean tensor of shape [num_edges], used to indicate which edges in the original graph are included in the subgraph.

        # GNN part
        x = self.entity_encoder(subset)  # [subset_entity_size, in_channels]
        x = self.gnn(x, sub_edge_index, sub_edge_type)

        # Index x by subscript to get corresponding gene node embeddings
        x = x[torch.searchsorted(subset, gene_entity, right=False)]  # [batch_size, max_seq_len, out_channels]

        return x


class SingleModalClassifier(nn.Module):
    def __init__(self, gene_final_dim):
        super().__init__()

        self.fc = nn.Linear(int(2 * gene_final_dim), 1)

    def forward(self, gene_1_emb, gene_2_emb):
        combined_emb = torch.cat((gene_1_emb, gene_2_emb), dim=1)
        combined_emb = self.fc(combined_emb)
        return combined_emb


class UMTClassifier(nn.Module):
    def __init__(self, gene_final_dim, final_mlp_dropout):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(int(4 * gene_final_dim), gene_final_dim),
            nn.ReLU(),
            nn.Dropout(final_mlp_dropout),
            nn.Linear(gene_final_dim, 1),
        )

    def forward(self, gene_1_distill_omics_emb, gene_2_distill_omics_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb):
        combined_emb = torch.cat((gene_1_distill_omics_emb, gene_2_distill_omics_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb), dim=1)
        combined_emb = self.fc(combined_emb)
        return combined_emb


class UMTModel(nn.Module):
    def __init__(self, omics_encoder_params, kg_encoder_params, gene_final_dim, final_mlp_dropout):
        super().__init__()
        self.pretrain_omics_encoder = OmicsEncoder(omics_encoder_params)
        self.distill_omics_encoder = OmicsEncoder(omics_encoder_params)
        self.pretrain_kg_encoder = KGEncoder(kg_encoder_params)
        self.distill_kg_encoder = KGEncoder(kg_encoder_params)

        self.omics_classifier = SingleModalClassifier(gene_final_dim)
        self.kg_classifier = SingleModalClassifier(gene_final_dim)
        self.umt_classifier = UMTClassifier(gene_final_dim, final_mlp_dropout)

    def forward(self, task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params):
        if task_type == "only_omics":
            gene_1_emb, gene_1_self_kl_loss, gene_1_cross_kl_loss = self.pretrain_omics_encoder(gene_1_omics_params)
            gene_2_emb, gene_2_self_kl_loss, gene_2_cross_kl_loss = self.pretrain_omics_encoder(gene_2_omics_params)
            logit = self.omics_classifier(gene_1_emb, gene_2_emb)
            return logit, (gene_1_self_kl_loss + gene_2_self_kl_loss), (gene_1_cross_kl_loss + gene_2_cross_kl_loss)

        elif task_type == "only_kg":
            logit = self.kg_classifier(self.pretrain_kg_encoder(gene_1_kg_params), self.pretrain_kg_encoder(gene_2_kg_params))
            return logit

        elif task_type == "umt":
            is_training = gene_1_omics_params["is_training"]  # Whether in training
            # pretrain
            gene_1_omics_params["is_training"] = False  # Pre-trained model directly uses mean in OmicsEncoder
            gene_2_omics_params["is_training"] = False
            gene_1_pretrain_omics_emb, _, _ = self.pretrain_omics_encoder(gene_1_omics_params)
            gene_2_pretrain_omics_emb, _, _ = self.pretrain_omics_encoder(gene_2_omics_params)
            gene_1_pretrain_kg_emb = self.pretrain_kg_encoder(gene_1_kg_params)
            gene_2_pretrain_kg_emb = self.pretrain_kg_encoder(gene_2_kg_params)

            # scratch
            gene_1_omics_params["is_training"] = is_training  # Restore original value
            gene_2_omics_params["is_training"] = is_training
            gene_1_distill_omics_emb, gene_1_self_kl_loss, gene_1_cross_kl_loss = self.distill_omics_encoder(gene_1_omics_params)
            gene_2_distill_omics_emb, gene_2_self_kl_loss, gene_2_cross_kl_loss = self.distill_omics_encoder(gene_2_omics_params)
            gene_1_distill_kg_emb = self.distill_kg_encoder(gene_1_kg_params)
            gene_2_distill_kg_emb = self.distill_kg_encoder(gene_2_kg_params)

            # concat
            combined_pretrain_omics_emb = torch.cat((gene_1_pretrain_omics_emb, gene_2_pretrain_omics_emb), dim=1)
            combined_distill_omics_emb = torch.cat((gene_1_distill_omics_emb, gene_2_distill_omics_emb), dim=1)
            combined_pretrain_kg_emb = torch.cat((gene_1_pretrain_kg_emb, gene_2_pretrain_kg_emb), dim=1)
            combined_distill_kg_emb = torch.cat((gene_1_distill_kg_emb, gene_2_distill_kg_emb), dim=1)

            # logit
            logit = self.umt_classifier(gene_1_distill_omics_emb, gene_2_distill_omics_emb, gene_1_distill_kg_emb, gene_2_distill_kg_emb)

            return logit, combined_pretrain_omics_emb, combined_distill_omics_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, (gene_1_self_kl_loss + gene_2_self_kl_loss), (gene_1_cross_kl_loss + gene_2_cross_kl_loss)

        elif task_type == "ume":
            gene_1_omics_emb, _, _ = self.pretrain_omics_encoder(gene_1_omics_params)
            gene_2_omics_emb, _, _ = self.pretrain_omics_encoder(gene_2_omics_params)
            logit_omics = self.omics_classifier(gene_1_omics_emb, gene_2_omics_emb)
            logit_kg = self.kg_classifier(self.pretrain_kg_encoder(gene_1_kg_params), self.pretrain_kg_encoder(gene_2_kg_params))
            return logit_omics, logit_kg

    @staticmethod
    def load_state_dicts(model, dicts):
        for module_name, state_dict in dicts.items():
            getattr(model, module_name).load_state_dict(state_dict)

    @staticmethod
    def get_target_module_names(task_type):
        module_names = None
        if task_type == "only_omics":
            module_names = ["pretrain_omics_encoder", "omics_classifier"]
        elif task_type == "only_kg":
            module_names = ["pretrain_kg_encoder", "kg_classifier"]
        elif task_type == "umt":
            module_names = ["pretrain_omics_encoder", "distill_omics_encoder", "pretrain_kg_encoder", "distill_kg_encoder", "umt_classifier"]
        elif task_type == "ume":
            module_names = []  # ume does not save the model
        return module_names

    @staticmethod
    def state_dicts(model, task_type, accelerator):
        model = accelerator.unwrap_model(model)
        module_names = UMTModel.get_target_module_names(task_type)
        dicts = {}
        for module_name in module_names:
            state_dict = getattr(model, module_name).state_dict()
            dicts[module_name] = state_dict
        return dicts


class MultiOptimizer:
    def __init__(self, model, lr, weight_decay):
        optimizers = {}
        for module_name, module in model.named_children():
            optimizers[module_name] = MultiOptimizer.__create_optimizer(module, lr, weight_decay)
        self.optimizers = optimizers

    def load_state_dicts(self, dicts):  # Execute before prepare
        for module_name, state_dict in dicts.items():
            self.optimizers[module_name].load_state_dict(state_dict)

    def prepare(self, accelerator):
        for module_name, optimizer in self.optimizers.items():
            optimizer = accelerator.prepare(optimizer)
            self.optimizers[module_name] = optimizer

    def zero_grad(self, task_type):
        module_names = UMTModel.get_target_module_names(task_type)
        for module_name in module_names:
            self.optimizers[module_name].zero_grad()

    def step(self, task_type):
        module_names = UMTModel.get_target_module_names(task_type)
        for module_name in module_names:
            self.optimizers[module_name].step()

    def state_dicts(self, task_type, accelerator):
        module_names = UMTModel.get_target_module_names(task_type)
        dicts = {}
        for module_name in module_names:
            state_dict = accelerator.unwrap_model(self.optimizers[module_name]).state_dict()
            dicts[module_name] = state_dict
        return dicts

    @staticmethod
    def __create_optimizer(module, lr, weight_decay):
        """
        @From: https://github.com/SIAT-code/MASSA
        """

        # Randomly initialize model weights
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # weight: L2 regularization; bias: not L2 regularization
        weight, bias = [], []
        for name, param in module.named_parameters():
            if "bias" in name:
                bias.append(param)
            else:
                weight.append(param)

        optimizer_inner = RAdam(
            [
                {"params": weight, "weight_decay": weight_decay, "lr": lr},
                {"params": bias, "weight_decay": 0.0, "lr": lr},
            ],
            betas=(0.9, 0.999),  # default
            eps=1e-8,  # default
        )

        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)  # default
        return optimizer


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, accelerator, patience=7, verbose=False, reverse=False, delta=0):
        """
        Args:
            accelerator: Process
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            reverse (bool): True means smaller is better, False means larger is better
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse
        self.counter = 0
        self.best_score = None
        self.early_stop = False  # whether the model has already early stopped.
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        if self.reverse:
            score = -val_loss  # loss, small better
        else:
            score = val_loss  # AUC/AUPR, large better

        if self.best_score is None:
            self.best_score = score
            self.update_val_loss_min(val_loss)
            return True
        elif score <= self.best_score + self.delta:  # not improved
            self.counter += 1

            if self.verbose:
                rank_main_print(self.accelerator, f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:  # improved
            self.best_score = score
            self.update_val_loss_min(val_loss)
            self.counter = 0
            return True

    def update_val_loss_min(self, val_loss):
        if self.verbose:
            rank_main_print(self.accelerator, f"The validation indicator is better ({self.val_loss_min:.6f} --> {val_loss:.6f}).")

        self.val_loss_min = val_loss
