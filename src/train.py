import argparse
import copy
from datetime import datetime
import json
import os
import time

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate import DistributedDataParallelKwargs as DDPK
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MultiModalDataset, SLDataset
from executor import Tester, Trainer
from model import EarlyStopping, MultiOptimizer, UMTModel
from util.my import rank_main_print, rank_print, set_seed, time_elapsed


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # program
    parser.add_argument("--device_id_for_debug", type=int, default=-1, help="a single GPU device id when debugging. MUST set -1 when using accelerate")
    parser.add_argument("--specify_result_saving_folder", type=str, default="", help="save current result to specified folder if the value is not empty")

    parser.add_argument("--task_type", type=str, default="only_omics", help="only_omics, only_kg, umt, ume")
    parser.add_argument("--omics_ckpt_path", type=str, default="", help="omics teacher checkpoint data path")
    parser.add_argument("--kg_ckpt_path", type=str, default="", help="kg teacher checkpoint data path")

    # model
    parser.add_argument("--cancer_type", type=str, default="BRCA", help="one cancer type")
    parser.add_argument("--metric", type=int, default=1, help="CV1, CV2, CV3")
    parser.add_argument("--train_fold", type=int, default=1, help="one of the folds in 5-fold cross validation")
    parser.add_argument("--epochs", type=int, default=200, help="number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=384, help="batch size of training, validating and testing")
    parser.add_argument("--split_batches", type=bool, default=True, help="If True, the actual batch size of model is `batch_size`, which must be an integer multiple of GPUs count you used. If False, the actual batch size of model is `batch_size` * `number of GPUs`.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularized weight decay factor")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="gradient accumulation quantity")
    parser.add_argument("--patience", type=int, default=999, help="the number of epoch of early stop tolerance")

    # OmicsEncoder
    parser.add_argument("--hid_dim", type=int, default=128, help="hidden dim after Encoder")
    parser.add_argument("--omics_types", nargs="+", type=str, default=["cna", "exp", "mut", "myl"], help="name of omics category used")
    parser.add_argument("--vae_hidden_dims", nargs="+", type=int, default=[512, 256], help="hidden dims list of VAE")
    parser.add_argument("--vae_dropout", type=float, default=0.2, help="dropout rate of VAE layer")
    parser.add_argument("--self_kl_loss_weight", type=float, default=0.1)
    parser.add_argument("--cross_kl_loss_weight", type=float, default=0.5)

    # KGEncoder
    parser.add_argument("--in_channels", type=int, default=128, help="input dimension of RGCN")
    parser.add_argument("--hidden_channels", type=int, default=256, help="hidden dimension of RGCN")
    parser.add_argument("--gcn_layers", type=int, default=2, help="layers count of RGCN")
    parser.add_argument("--graph_dropout", type=float, default=0.5, help="dropout rate of the graph")

    # Classifier
    parser.add_argument("--gene_final_dim", type=int, default=128, help="gene final dim to input classifier")
    parser.add_argument("--final_mlp_dropout", type=float, default=0.5, help="the dropout of final dimensionality reduction MLP")
    parser.add_argument("--lambda_distill", type=float, default=1, help="weight of distillation loss")

    return parser.parse_args()


def load_checkpoint(ckpt_path, accelerator, model, multi_optimizer):
    checkpoint_dict = torch.load(ckpt_path, weights_only=True, map_location=accelerator.device)
    model_dicts = checkpoint_dict["model"]
    optimizer_dicts = checkpoint_dict["op"]
    UMTModel.load_state_dicts(model, model_dicts)
    multi_optimizer.load_state_dicts(optimizer_dicts)


def main():
    start_time_prepare = time.time()

    args = init_argparse()

    kwargs = DDPK(find_unused_parameters=True, broadcast_buffers=True)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=args.gradient_accumulation,
        dataloader_config=DataLoaderConfiguration(split_batches=args.split_batches),
    )
    rank_main_print(accelerator, f"param: {args}")

    # -1 means no intervention with accelerator, currently GPU0. Can also set other values during debug
    if accelerator.is_main_process:  # Only main process operates
        if args.device_id_for_debug != -1:
            torch.cuda.set_device(args.device_id_for_debug)

    # Set random seed
    set_seed(2025)

    ###################################################################################### result related settings
    result_fold_save_path_prefix = "../result"
    result_folder_name = args.specify_result_saving_folder
    if result_folder_name.strip() == "":
        current_time = datetime.now()
        result_folder_name = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    result_fold_path = result_fold_save_path_prefix + "/" + result_folder_name

    if accelerator.is_main_process:
        if not os.path.exists(result_fold_path):
            os.makedirs(result_fold_path)

        # Save hyperparameters
        with open(result_fold_path + "/hyper_parameters.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        print("success to save hyper parameters to:", result_fold_path + "/hyper_parameters.json")

    ###################################################################################### create model
    # read dataset
    data_base_path = f"../data/{args.cancer_type}"
    mm_dataset = MultiModalDataset(
        accelerator,
        gene_path=os.path.join(data_base_path, "gene.pt"),
        omics_path_dict={omics_type: f"{data_base_path}/{omics_type}.npy" for omics_type in args.omics_types},
        kg_graph_path=os.path.join(data_base_path, "kg.pt"),
    )
    # Label data
    sl_train_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/train_sl.npy", mm_dataset=mm_dataset)
    sl_val_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/val_sl.npy", mm_dataset=mm_dataset)
    sl_test_dataset = SLDataset(accelerator, sl_path=f"../data/{args.cancer_type}/cv_{args.metric}_fold_{args.train_fold}/test_sl.npy", mm_dataset=mm_dataset)

    omics_encoder_params = {
        "omics_count": len(args.omics_types),
        "omics_input_dim": mm_dataset.get_omics_input_dim(),
        "vae_hidden_dims": args.vae_hidden_dims,
        "vae_dropout": args.vae_dropout,
        "hid_dim": args.hid_dim,
    }
    kg_encoder_params = {
        "entity_vocab_size": mm_dataset.get_entity_vocab_size(),
        "hid_dim": args.hid_dim,
        "in_channels": args.in_channels,
        "hidden_channels": args.hidden_channels,
        "gcn_layers": args.gcn_layers,
        "graph_dropout": args.graph_dropout,
        "num_relations": mm_dataset.get_kg_relations_count(),
    }
    model = UMTModel(
        omics_encoder_params=omics_encoder_params,
        kg_encoder_params=kg_encoder_params,
        gene_final_dim=args.gene_final_dim,
        final_mlp_dropout=args.final_mlp_dropout,
    )

    early_stopping = EarlyStopping(accelerator, patience=args.patience, verbose=True, reverse=False)
    multi_optimizer = MultiOptimizer(model, args.lr, args.weight_decay)

    # Load teacher model
    task_type = args.task_type
    if task_type == "umt" or task_type == "ume":
        load_checkpoint(args.omics_ckpt_path, accelerator, model, multi_optimizer)
        load_checkpoint(args.kg_ckpt_path, accelerator, model, multi_optimizer)

        # Freeze parameters
        for param in model.pretrain_omics_encoder.parameters():
            param.requires_grad = False
        for param in model.pretrain_kg_encoder.parameters():
            param.requires_grad = False

    train_dataloader = DataLoader(sl_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(sl_val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(sl_test_dataset, batch_size=args.batch_size)

    model, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)
    multi_optimizer.prepare(accelerator)

    lambda_distill = args.lambda_distill
    trainer = Trainer(model, args.batch_size, multi_optimizer, lambda_distill=lambda_distill, self_kl_loss_weight=args.self_kl_loss_weight, cross_kl_loss_weight=args.cross_kl_loss_weight)
    validator = Tester(model, args.batch_size, lambda_distill=lambda_distill, self_kl_loss_weight=args.self_kl_loss_weight, cross_kl_loss_weight=args.cross_kl_loss_weight)
    tester = Tester(model, args.batch_size, lambda_distill=lambda_distill, self_kl_loss_weight=args.self_kl_loss_weight, cross_kl_loss_weight=args.cross_kl_loss_weight)

    end_time_prepare = time.time()

    ###################################################################################### Start training
    start_time_learn = time.time()

    kg_graph = mm_dataset.get_kg_graph()

    df_evaluate = pd.DataFrame()  # Evaluation data for train and val
    checkpoint_epoch = 0
    checkpoint_model = None
    checkpoint_optimizer = None

    for epoch in range(1, args.epochs + 1):
        start_time_epoch = time.time()
        rank_main_print(accelerator, f"\n###################################################################################### training epoch: {epoch}")

        # Training
        rank_main_print(accelerator, "training...")
        if task_type == "ume":  # ume does not perform training
            evaluate_results_train = {"AUC": 0, "AUPR": 0, "F1": 0}
        else:
            evaluate_results_train = trainer.train(accelerator, train_dataloader, kg_graph, task_type)
        rank_main_print(accelerator, f"evaluate_results_train: {evaluate_results_train}")

        # Validating
        rank_main_print(accelerator, "validating...")
        evaluate_results_val = validator.test(accelerator, val_dataloader, kg_graph, task_type)
        val_auc = evaluate_results_val["AUC"]  # AUC of validation set, used for subsequent early stopping
        rank_main_print(accelerator, f"evaluate_results_val: {evaluate_results_val}")

        rank_main_print(accelerator, "testing...")
        evaluate_results_test = tester.test(accelerator, test_dataloader, kg_graph, task_type)
        rank_main_print(accelerator, f"evaluate_results_test: {evaluate_results_test}")

        # Statistics
        if accelerator.is_main_process:
            evaluate_results_train = {f"train_{key}": value for key, value in evaluate_results_train.items()}
            evaluate_results_val = {f"val_{key}": value for key, value in evaluate_results_val.items()}
            evaluate_results_test = {f"test_{key}": value for key, value in evaluate_results_test.items()}
            evaluate_results_merge = evaluate_results_train | evaluate_results_val | evaluate_results_test
            evaluate_results_merge["epoch"] = epoch

            df_evaluate = pd.concat([df_evaluate, pd.DataFrame([evaluate_results_merge])], ignore_index=True)

        # Print time elapsed
        end_time_epoch = time.time()
        rank_main_print(accelerator, f"epoch {epoch} time use: {time_elapsed(start_time_epoch, end_time_epoch)}")

        # Early stopping, every process will execute once
        is_result_better = early_stopping(val_auc)
        if is_result_better:
            accelerator.wait_for_everyone()  # Wait
            if accelerator.is_main_process:  # Only main process operates
                checkpoint_model = UMTModel.state_dicts(model, args.task_type, accelerator)
                checkpoint_optimizer = multi_optimizer.state_dicts(args.task_type, accelerator)
                # Clone
                checkpoint_model = copy.deepcopy(checkpoint_model)
                checkpoint_optimizer = copy.deepcopy(checkpoint_optimizer)
                checkpoint_epoch = epoch

        if task_type != "ume" and early_stopping.early_stop:
            rank_main_print(accelerator, "Early Stopping!!!")
            break

        if task_type == "ume":  # ume only executes 1 epoch
            break

    # Save model file
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.task_type != "ume":
            torch.save(
                {"epoch": checkpoint_epoch, "model": checkpoint_model, "op": checkpoint_optimizer},
                f"{result_fold_path}/checkpoint.pth",
            )
            print("success to save model checkpoint to:", f"{result_fold_path}/checkpoint.pth")

    # Main process performs other operations
    if accelerator.is_main_process:
        df_evaluate.to_csv(result_fold_path + "/train_val_evaluate.csv", index=False)
        print("success to save train and val evaluate data to:", result_fold_path + "/train_val_evaluate.csv")

    # show ending
    end_time_learn = time.time()
    ending_str = "Finished!!! " + time_elapsed(start_time_prepare, end_time_prepare, "prepare time use: ") + ", " + time_elapsed(start_time_learn, end_time_learn, "learn time use: ") + "."
    rank_print(accelerator, ending_str)


if __name__ == "__main__":
    main()
