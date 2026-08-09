import os
import glob
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.der import DER
from utils.data_manager import DataManager
from utils.toolkit import calculate_metrics
from utils.inc_net import DERNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    args = {
        "dataset": "cic_iot23",
        "memory_size": 5000,
        "memory_per_class": 20,
        "fixed_memory": False,
        "shuffle": False,
        "init_cls": 6,
        "increment": 6,
        "model_name": "der",
        "convnet_type": "cnn1d",
        "device": [device],
        "seed": 42,
        "batch_size": 1024,
        "num_workers": 0
    }

    # Load existing results to avoid recomputing
    existing_results = {}
    existing_csv = "logs/der/cic_iot23/test_results (7).csv"
    if os.path.exists(existing_csv):
        df_exist = pd.read_csv(existing_csv)
        for _, row in df_exist.iterrows():
            ckpt = row['checkpoint']
            match = re.search(r"ckpt_task(\d+)_round(\d+)\.pth", ckpt)
            if match:
                t_id = int(match.group(1))
                r_id = int(match.group(2))
                # Map old columns to requested format
                res = {
                    "accuracy": float(row["acc"]),
                    "f1_micro": float(row["f1_mic"]),
                    "f1_macro": float(row["f1_mac"]),
                    "f1_weight": float(row["f1_wei"]),
                    "precision_micro": float(row["prec_mic"]),
                    "precision_macro": float(row["prec_mac"]),
                    "precision_weight": float(row["prec_wei"]),
                    "recall_micro": float(row["rec_mic"]),
                    "recall_macro": float(row["rec_mac"]),
                    "recall_weight": float(row["rec_wei"]),
                    "loss": float(row["loss"])
                }
                existing_results[(t_id, r_id)] = res
    print(f"Loaded {len(existing_results)} existing checkpoint metrics.", flush=True)

    # Initialize model if we need to compute missing
    data_manager = None
    model = None

    ckpt_dir = "logs/der/cic_iot23"
    all_ckpts = []
    for root, dirs, files in os.walk(ckpt_dir):
        if "checkpoints" in root:
            for file in files:
                if file.startswith("ckpt_task") and "_round" in file and file.endswith(".pth"):
                    match = re.search(r"ckpt_task(\d+)_round(\d+)\.pth", file)
                    if match:
                        task_id = int(match.group(1))
                        round_id = int(match.group(2))
                        full_path = os.path.join(root, file)
                        all_ckpts.append((task_id, round_id, full_path))
    
    all_ckpts.sort(key=lambda x: (x[0], x[1]))
    
    valid_ckpts = []
    for task in range(6):
        for rnd in range(1, 31):
            c = [x for x in all_ckpts if x[0] == task and x[1] == rnd]
            if c:
                valid_ckpts.append(c[-1])
                
    print(f"Total valid checkpoints found: {len(valid_ckpts)}", flush=True)
    
    results = []
    criterion = torch.nn.CrossEntropyLoss()
    
    for task_id, round_id, ckpt_path in valid_ckpts:
        if (task_id, round_id) in existing_results:
            results.append(existing_results[(task_id, round_id)])
            continue
            
        if data_manager is None:
            print("Loading data manager for missing checkpoints...", flush=True)
            data_manager = DataManager(
                args["dataset"], args["shuffle"], args["seed"],
                args["init_cls"], args["increment"]
            )
            model = DER(args)
            
        print(f"Evaluating missing Task {task_id}, Round {round_id} from {ckpt_path}...", flush=True)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint["model_state_dict"]
        
        model._cur_task = task_id
        model._known_classes = sum(data_manager.get_task_size(i) for i in range(task_id))
        model._total_classes = model._known_classes + data_manager.get_task_size(task_id)
        
        model._network = DERNet(args, False)
        model._network.update_fc(model._total_classes)
        for _ in range(task_id):
            model._network.update_fc(model._total_classes)
            
        model._network.load_state_dict(model_state)
        model._network.to(device)
        model._network.eval()
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, model._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"]
        )
        
        y_pred, y_true = [], []
        total_loss, num_samples = 0.0, 0
        
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                outputs = model._network(inputs)["logits"]
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
                predicts = torch.max(outputs, dim=1)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
                
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        avg_loss = total_loss / max(1, num_samples)
        
        metrics = calculate_metrics(y_true, y_pred)
        acc = metrics["total"]
        prec_mic = metrics["precision_micro"]
        prec_mac = metrics["precision_macro"]
        prec_wei = metrics["precision_weighted"]
        rec_mic = metrics["recall_micro"]
        rec_mac = metrics["recall_macro"]
        rec_wei = metrics["recall_weighted"]
        f1_mic = metrics["f1_micro"]
        f1_mac = metrics["f1_macro"]
        f1_wei = metrics["f1_weighted"]
        
        res = {
            "accuracy": np.round(acc, 2),
            "f1_micro": np.round(f1_mic, 2),
            "f1_macro": np.round(f1_mac, 2),
            "f1_weight": np.round(f1_wei, 2),
            "precision_micro": np.round(prec_mic, 2),
            "precision_macro": np.round(prec_mac, 2),
            "precision_weight": np.round(prec_wei, 2),
            "recall_micro": np.round(rec_mic, 2),
            "recall_macro": np.round(rec_mac, 2),
            "recall_weight": np.round(rec_wei, 2),
            "loss": np.round(avg_loss, 4)
        }
        results.append(res)
        print(f"  -> Acc: {acc:.2f}%, Loss: {avg_loss:.4f}", flush=True)
        
    df_all = pd.DataFrame(results)
    
    # Save the 180 rounds file
    df_all.to_csv("spcil_180_rounds.csv", index=False)
    print("Saved spcil_180_rounds.csv", flush=True)
    
    # Extract only round 30 for each task.
    # Since they are ordered chronologically and there are 30 rounds per task
    final_rounds = []
    for i in range(29, len(df_all), 30):
        final_rounds.append(df_all.iloc[i])
        
    df_final = pd.DataFrame(final_rounds)
    df_final.to_csv("spcil_final_rounds.csv", index=False)
    print("Saved spcil_final_rounds.csv", flush=True)

if __name__ == "__main__":
    main()
