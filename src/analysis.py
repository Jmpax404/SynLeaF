import argparse
import numpy as np
import pandas as pd


def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--task_folder_prex", type=str, default="default", help="the prefix of task folder to distinguish different configurations")
    parser.add_argument(
        "--cancers",
        nargs="+",
        type=str,
        default=["BRCA", "CESC", "COAD", "KIRC", "LAML", "LUAD", "OV", "SKCM", "pan"],
        help="one or more cancer types",
    )
    parser.add_argument("--is_print_fold", type=int, default=0, help="1: print, 0: not")

    return parser.parse_args()


def cal(cancer, cv, model_name, task_folder_prex, is_print_fold=False):
    data = []
    val_aucs = []

    for fold in range(1, 5 + 1):
        try:
            train_val_evaluate_path = f"../result/{task_folder_prex}_{cancer}_cv_{cv}_fold_{fold}_{model_name}/train_val_evaluate.csv"
            df = pd.read_csv(train_val_evaluate_path)
            max_auc_row = df.loc[df["val_AUC"].idxmax()]

            test_auc = max_auc_row["test_AUC"]
            test_aupr = max_auc_row["test_AUPR"]
            test_f1 = max_auc_row["test_F1"]

            data.append([test_auc, test_aupr, test_f1])
            val_aucs.append(max_auc_row["val_AUC"])
        except FileNotFoundError:
            print(f"Fold {fold}: not exist.")
            data.append([0, 0, 0])
            val_aucs.append(0)

    if is_print_fold:
        print("\t")
        for fold in range(1, 5 + 1):
            try:
                print(f"{fold}, {data[fold-1][0]:.4f}, {data[fold-1][1]:.4f}, {data[fold-1][2]:.4f}")
            except FileNotFoundError:
                pass

    data_array = np.array(data)
    means = np.mean(data_array, axis=0)
    stds = np.std(data_array, axis=0)

    print(f"{means[0]:.4f}({stds[0]:.4f}), {means[1]:.4f}({stds[1]:.4f}), {means[2]:.4f}({stds[2]:.4f}) {task_folder_prex} {cancer} cv{cv} {model_name}")
    return np.mean(val_aucs) if val_aucs else 0


def main():
    args = init_argparse()
    task_folder_prex = args.task_folder_prex
    cancers = args.cancers
    is_print_fold = args.is_print_fold == 1

    for cancer in cancers:
        cv_s = [1, 2]
        if cancer == "pan":
            cv_s.append(3)

        for cv in cv_s:
            print(f"\n# {cancer} cv{cv}")

            cal(cancer, cv, "only_omics", task_folder_prex, is_print_fold)
            cal(cancer, cv, "only_kg", task_folder_prex, is_print_fold)
            umt_val_auc = cal(cancer, cv, "umt", task_folder_prex, is_print_fold)
            ume_val_auc = cal(cancer, cv, "ume", task_folder_prex, is_print_fold)

            final_use = "UMT" if umt_val_auc > ume_val_auc else "UME"
            print(f"val_AUC_UMT={umt_val_auc:.4f}, val_AUC_UME={ume_val_auc:.4f}, use {final_use}.")


if __name__ == "__main__":
    main()
