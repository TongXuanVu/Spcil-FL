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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dummy args
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
        "num_workers": 4
    }

    # Load DataManager
    print("Loading data manager...")
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"]
    )
    
    # Initialize Model
    model = DER(args)
    
    # Find all checkpoints
    ckpt_dir = "logs/der/cic_iot23"
    all_ckpts = []
    
    # Find files recursively
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
    
    # Sort checkpoints by task, then round
    all_ckpts.sort(key=lambda x: (x[0], x[1]))
    
    # We should have exactly 180 checkpoints
    print(f"Found {len(all_ckpts)} checkpoints!")
    
    # Ensure there are no duplicates and we have everything
    valid_ckpts = []
    for task in range(6):
        for rnd in range(1, 31):
            # Find the checkpoint
            c = [x for x in all_ckpts if x[0] == task and x[1] == rnd]
            if not c:
                print(f"MISSING: Task {task}, Round {rnd}")
            else:
                # If there are duplicates, take the one from the newer log folder (which is later in sorting usually)
                valid_ckpts.append(c[-1])
                
    print(f"Ready to evaluate {len(valid_ckpts)} valid checkpoints.")
    
    results = []
    
    # Evaluation loop
    criterion = torch.nn.CrossEntropyLoss()
    
    for task_id, round_id, ckpt_path in valid_ckpts:
        print(f"Evaluating Task {task_id}, Round {round_id} from {ckpt_path}...")
        
        # Load Checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint["model_state_dict"]
        
        # Setup model architecture to match checkpoint
        model._cur_task = task_id
        model._known_classes = sum(data_manager.get_task_size(i) for i in range(task_id))
        model._total_classes = model._known_classes + data_manager.get_task_size(task_id)
        
        # Build incremental network structure without training
        model._network = model.network_class(args["convnet_type"], False)
        model._network.update_fc(model._total_classes)
        for _ in range(task_id):
            model._network.update_fc(model._total_classes)
            
        model._network.load_state_dict(model_state)
        model._network.to(device)
        model._network.eval()
        
        # Load Test Data for current task
        test_dataset = data_manager.get_dataset(
            np.arange(0, model._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"]
        )
        
        # Evaluate CNN
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
        
        # Calculate Metrics
        metrics = calculate_metrics(y_true, y_pred)
        acc = metrics["total"]
        prec = metrics["precision_macro"]
        rec = metrics["recall_macro"]
        f1 = metrics["f1_macro"]
        
        results.append({
            "task_id": task_id,
            "round": round_id,
            "accuracy": np.round(acc, 2),
            "precision": np.round(prec, 2),
            "recall": np.round(rec, 2),
            "f1_score": np.round(f1, 2),
            "loss": np.round(avg_loss, 4)
        })
        print(f"  -> Acc: {acc:.2f}%, Loss: {avg_loss:.4f}")
        
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("evaluation_spcil_cic_iot23.csv", index=False)
    print("Evaluation completed. Saved to evaluation_spcil_cic_iot23.csv")

if __name__ == "__main__":
    main()
