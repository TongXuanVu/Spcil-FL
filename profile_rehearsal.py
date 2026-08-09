import sys
import os
import time
import torch
import numpy as np

sys.path.append(r"C:\FederatedLearning\SPCIL")
from utils.data_manager import DataManager
from models.der import DER

def profile_rehearsal():
    print("Loading data manager...")
    args = {
        "dataset": "cic_iot23",
        "shuffle": False,
        "seed": 42,
        "init_cls": 6,
        "increment": 6,
        "model_name": "der",
        "convnet_type": "cnn1d",
        "device": ["cpu"],
        "batch_size": 512
    }
    
    t0 = time.time()
    data_manager = DataManager("cic_iot23", False, 42, 6, 6)
    print(f"Data manager loaded in {time.time()-t0:.2f}s")
    
    model = DER(args)
    model._network.to("cpu")
    
    print("Setting up network architecture...")
    for t in range(4):
        model.incremental_train(data_manager, skip_train=True)
        model.after_task()
    
    print("Starting build_rehearsal_memory for Tasks 0 to 3...")
    t0 = time.time()
    for t in range(4):
        model._cur_task = t
        model._known_classes = sum(data_manager.get_task_size(i) for i in range(t))
        model._total_classes = model._known_classes + data_manager.get_task_size(t)
        model.build_rehearsal_memory(data_manager, model.samples_per_class)
        print(f"Task {t} completed. Memory size: {len(model._data_memory)}")
    t1 = time.time()
    
    print(f"Full resume rebuild took {t1 - t0:.2f} seconds!")

if __name__ == "__main__":
    profile_rehearsal()
