"""SPCIL-FL — điểm vào bản federated (FedAvg).

    python main_fl.py --config exps_fl/cic_iot23_fl_fewshot1.json \
        --data_root "<.../100client>" \
        --fewshot_dir "<.../federated_data_fewshot>"

Yêu cầu: code SPCIL gốc nằm cùng thư mục (xem README_FL.md, mục 1).
`main.py` và `trainer.py` của SPCIL không bị đụng tới — bản tập trung chạy như cũ.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl.trainer_fl import train


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_parser():
    p = argparse.ArgumentParser(description="SPCIL-FL — federated (FedAvg)")
    p.add_argument("--config", type=str, default="./exps_fl/cic_iot23_fl.json")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint (.pth) de tiep tuc. Nen dung file *_FINAL.pth.")
    p.add_argument("--data_root", type=str, default=None,
                   help="Thu muc chua client_*_task_*.pt. Bo trong = tu do.")
    p.add_argument("--fewshot_dir", type=str, default=None,
                   help="Thu muc du lieu few-shot (task 2..6). Bo trong = full data.")
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--memory_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="So epoch CUC BO moi round.")
    p.add_argument("--debug", action="store_true",
                   help="Chay thu: 5 client, 2 round, 1 epoch.")
    return p


def main():
    cli = setup_parser().parse_args()
    args = load_json(cli.config)
    args.update({k: v for k, v in vars(cli).items() if v is not None})

    if args.get("debug"):
        print("[DEBUG] num_clients=5, num_rounds=2, epochs=1")
        args["epochs"] = 1
        args["init_epoch"] = 1
        args["num_rounds"] = 2
        args["num_clients"] = 5

    train(args)


if __name__ == "__main__":
    main()
