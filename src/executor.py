import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm


# Binary Cross-Entropy (Task Loss)
def task_loss_function(predictions, labels):
    return nn.BCEWithLogitsLoss()(predictions, labels)


# Mean Squared Error (Distillation Loss)
def distill_loss_function(student_features, teacher_features, temperature=1.0):
    return nn.MSELoss()(student_features / temperature, teacher_features / temperature)


def evaluate_performance(label, pred):
    """
    Evaluate SL task performance
    :param label: True labels, numpy
    :param pred: Predicted values, numpy
    """

    auc = roc_auc_score(label, pred)
    aupr = average_precision_score(label, pred)

    precision, recall, _ = precision_recall_curve(label, pred)
    denominator = precision + recall
    f1_scores = np.zeros_like(denominator)
    valid_mask = denominator > 0
    f1_scores[valid_mask] = 2 * (precision[valid_mask] * recall[valid_mask]) / denominator[valid_mask]
    f1 = np.max(f1_scores)

    performance_dict = {"AUC": auc, "AUPR": aupr, "F1": f1}
    return performance_dict


class Losser(object):
    """
    Loss statistician
    """

    def __init__(self, accelerator, all_batch_count):
        self.accelerator = accelerator
        self.all_batch_count = torch.tensor(all_batch_count).to(accelerator.device)  # Batch count per process
        self.all_loss_dict = {}

    def multi_incr(self, value_dict):
        for key, value in value_dict.items():
            if key not in self.all_loss_dict:
                self.all_loss_dict[key] = torch.tensor(0.0).to(self.accelerator.device)
            self.all_loss_dict[key] += value

    def get_results(self):
        """
        Reduce across processes, equivalent to dividing total loss by total batch count
        Must be done by every process
        """

        all_batch_count_reduce = self.accelerator.reduce(self.all_batch_count)

        results = {}
        for key, value in self.all_loss_dict.items():
            value_reduce = self.accelerator.reduce(value)
            value_per_batch = value_reduce / all_batch_count_reduce
            results[key] = value_per_batch.item()  # Extract scalar value from tensor

        return results


class Trainer(object):

    def __init__(self, model, batch_size, multi_optimizer, lambda_distill, self_kl_loss_weight, cross_kl_loss_weight):
        """
        :param model: Overall model
        :param batch_size: batch_size
        :param multi_optimizer: Custom multi-optimizer
        :param lambda_distill: Distillation weight
        :param self_kl_loss_weight: Used by omics
        :param cross_kl_loss_weight: Used by omics
        """

        self.model = model
        self.batch_size = batch_size
        self.multi_optimizer = multi_optimizer  # Optimizer
        self.lambda_distill = lambda_distill
        self.self_kl_loss_weight = self_kl_loss_weight
        self.cross_kl_loss_weight = cross_kl_loss_weight

    def train(self, accelerator, sl_dataloader, kg_graph, task_type):
        # Start training
        self.model.train()

        dataloader = sl_dataloader

        losser = Losser(accelerator=accelerator, all_batch_count=len(dataloader))
        all_predicts = torch.tensor([], device=accelerator.device)  # Accumulate predicted values
        all_labels = torch.tensor([], device=accelerator.device)  # Accumulate true values

        self.multi_optimizer.zero_grad(task_type)
        with tqdm(dataloader, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:  # only printed in the main process of each machine
            for step, data in enumerate(tepoch):
                with accelerator.accumulate(self.model):  # accelerator implements gradient accumulation
                    # Get data
                    gene_1_omics_data_list = data["gene_1_omics_data_list"]
                    gene_1_gene_entity = data["gene_1_gene_entity"]

                    gene_2_omics_data_list = data["gene_2_omics_data_list"]
                    gene_2_gene_entity = data["gene_2_gene_entity"]

                    label = data["label"]

                    # Form parameters
                    gene_1_omics_params = {"omics_data_list": gene_1_omics_data_list, "is_training": True}
                    gene_2_omics_params = {"omics_data_list": gene_2_omics_data_list, "is_training": True}
                    gene_1_kg_params = {"gene_entity": gene_1_gene_entity, "kg_graph": kg_graph}
                    gene_2_kg_params = {"gene_entity": gene_2_gene_entity, "kg_graph": kg_graph}

                    # Call model
                    logit = None
                    loss = None
                    if task_type == "only_omics":
                        logit, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)  # [batch_size, 1] -> [batch_size], value at this point has not passed through sigmoid
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})
                    elif task_type == "only_kg":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "umt":
                        logit, combined_pretrain_omics_emb, combined_distill_omics_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)
                        distill_loss = distill_loss_function(combined_distill_omics_emb, combined_pretrain_omics_emb) + distill_loss_function(combined_distill_kg_emb, combined_pretrain_kg_emb)

                        loss = label_loss + self.lambda_distill * distill_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-distill": distill_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})

                    # Backward propagation
                    accelerator.backward(loss)
                    self.multi_optimizer.step(task_type)
                    self.multi_optimizer.zero_grad(task_type)

                    # Add evaluation values for this batch
                    logit = logit.sigmoid()  # Value after sigmoid is [0,1]
                    all_predicts = torch.cat((all_predicts, logit), dim=0)
                    all_labels = torch.cat((all_labels, label), dim=0)

        results = {}  # Various statistical results

        # collecting all predictions and labels together from all processes
        all_predicts_gather = accelerator.gather(all_predicts)
        all_labels_gather = accelerator.gather(all_labels)
        if accelerator.is_main_process:  # Only the main process of the master node operates
            results = evaluate_performance(all_labels_gather.detach().cpu().numpy(), all_predicts_gather.detach().cpu().numpy())

        # reduce across processes
        results_loss = losser.get_results()
        results = results | results_loss

        return results


class Tester(object):
    """
    used for validation and test sets
    """

    def __init__(self, model, batch_size, lambda_distill, self_kl_loss_weight, cross_kl_loss_weight):
        """
        :param model: Overall model
        :param batch_size: batch_size
        :param lambda_distill: Distillation weight
        :param self_kl_loss_weight: Used by omics
        :param cross_kl_loss_weight: Used by omics
        """

        self.model = model
        self.batch_size = batch_size
        self.lambda_distill = lambda_distill
        self.self_kl_loss_weight = self_kl_loss_weight
        self.cross_kl_loss_weight = cross_kl_loss_weight

    def test(self, accelerator, sl_dataloader, kg_graph, task_type):
        # Start testing
        self.model.eval()

        dataloader = sl_dataloader

        losser = Losser(accelerator=accelerator, all_batch_count=len(dataloader))
        all_predicts = torch.tensor([], device=accelerator.device)  # Accumulate predicted values
        all_labels = torch.tensor([], device=accelerator.device)  # Accumulate true values

        with tqdm(dataloader, unit="batch", disable=not accelerator.is_local_main_process) as tepoch:  # only printed in the main process of each machine
            for step, data in enumerate(tepoch):
                # Get data
                gene_1_omics_data_list = data["gene_1_omics_data_list"]
                gene_1_gene_entity = data["gene_1_gene_entity"]

                gene_2_omics_data_list = data["gene_2_omics_data_list"]
                gene_2_gene_entity = data["gene_2_gene_entity"]

                label = data["label"]

                # Form parameters
                gene_1_omics_params = {"omics_data_list": gene_1_omics_data_list, "is_training": False}
                gene_2_omics_params = {"omics_data_list": gene_2_omics_data_list, "is_training": False}
                gene_1_kg_params = {"gene_entity": gene_1_gene_entity, "kg_graph": kg_graph}
                gene_2_kg_params = {"gene_entity": gene_2_gene_entity, "kg_graph": kg_graph}

                # Call model
                with torch.no_grad():
                    logit = None
                    loss = None
                    if task_type == "only_omics":
                        logit, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)  # [batch_size, 1] -> [batch_size], value at this point has not passed through sigmoid
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})
                    elif task_type == "only_kg":
                        logit = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)

                        loss = label_loss
                        losser.multi_incr({"loss-label": label_loss})
                    elif task_type == "umt":
                        logit, combined_pretrain_omics_emb, combined_distill_omics_emb, combined_pretrain_kg_emb, combined_distill_kg_emb, self_kl_loss, cross_kl_loss = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)
                        logit = logit.view(-1)
                        label_loss = task_loss_function(logit, label)
                        distill_loss = distill_loss_function(combined_distill_omics_emb, combined_pretrain_omics_emb) + distill_loss_function(combined_distill_kg_emb, combined_pretrain_kg_emb)

                        loss = label_loss + self.lambda_distill * distill_loss + self.self_kl_loss_weight * self_kl_loss + self.cross_kl_loss_weight * cross_kl_loss
                        losser.multi_incr({"loss-all": loss, "loss-label": label_loss, "loss-distill": distill_loss, "loss-self-kl": self_kl_loss, "loss-cross-kl": cross_kl_loss})

                    elif task_type == "ume":
                        logit_omics, logit_kg = self.model(task_type, gene_1_omics_params, gene_2_omics_params, gene_1_kg_params, gene_2_kg_params)

                # Add evaluation values for this batch
                if task_type == "ume":
                    logit_omics = logit_omics.sigmoid()
                    logit_kg = logit_kg.sigmoid()
                    logit = (logit_omics + logit_kg) * 0.5
                else:
                    logit = logit.sigmoid()  # Value after sigmoid is [0,1]

                all_predicts = torch.cat((all_predicts, logit), dim=0)
                all_labels = torch.cat((all_labels, label), dim=0)

        # collecting all predictions and labels together from all processes
        all_predicts_gather = accelerator.gather(all_predicts)
        all_labels_gather = accelerator.gather(all_labels)
        # Every process calculate score because early stopping requires it
        results = evaluate_performance(all_labels_gather.detach().cpu().numpy(), all_predicts_gather.detach().cpu().numpy())

        # reduce across processes
        results_loss = losser.get_results()
        results = results | results_loss

        return results
