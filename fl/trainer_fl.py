"""SPCIL-FL — lớp federated (FedAvg) bọc quanh SPCIL, KHÔNG sửa một dòng nào của SPCIL.

Mô phỏng một tiến trình: mọi client chạy tuần tự trong cùng chương trình. Mỗi round
server nạp trọng số toàn cục vào từng client, client huấn luyện cục bộ, server lấy
trung bình có trọng số theo số mẫu. Đúng FedAvg (McMahan et al. 2017), dễ debug, dễ
lưu checkpoint và resume — cùng kiểu với AFSIC-IDS, HFIN, MalCL-FL trong dự án.

    python main_fl.py --config exps_fl/cic_iot23_fl.json
"""
import copy
import csv
import glob
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager

from fl import data_fl


def average_weights(w, weights=None):
    """FedAvg: theta = sum(n_k * theta_k) / sum(n_k). weights=None -> trung bình đều."""
    if weights is None:
        weights = [1.0] * len(w)
    tot = float(sum(weights))
    out = copy.deepcopy(w[0])
    for key in out.keys():
        ref = out[key]
        if not torch.is_floating_point(ref):
            # num_batches_tracked và các bộ đếm nguyên khác
            acc = 0.0
            for i in range(len(w)):
                acc += float(w[i][key]) * weights[i]
            out[key] = torch.tensor(round(acc / tot), dtype=ref.dtype)
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for i in range(len(w)):
            acc += w[i][key].double() * weights[i]
        out[key] = (acc / tot).to(ref.dtype)
    return out


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_device(args):
    args["device"] = [torch.device("cpu") if str(d) == "-1" else torch.device(f"cuda:{d}")
                      for d in args["device"]]


def train(args):
    seeds = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seeds:
        args["seed"] = seed
        args["device"] = device
        _train_federated(args)


def _train_federated(args):
    data_fl.register()          # gắn dataset FL vào DataManager, không sửa SPCIL

    num_clients = args["num_clients"]
    num_rounds = args["num_rounds"]

    stamp = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join("logs", "spcil_fl", args["dataset"],
                           f"{stamp}_seed{args['seed']}_{args['convnet_type']}"
                           f"_clients{num_clients}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(os.path.join(run_dir, "training.log"),
                                      encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

    _set_random(args.get("seed_torch", 1))
    _set_device(args)
    for k, v in args.items():
        logging.info(f"{k}: {v}")
    args["run_dir"] = run_dir
    logging.info(f"Run directory: {run_dir}")

    # ── Kịch bản dữ liệu ──────────────────────────────────────────────────────
    data_fl.set_data_root(args.get("data_root"))
    fs = args.get("fewshot_dir") or None
    data_fl.set_fewshot_dir(fs)
    if fs:
        n = len(glob.glob(os.path.join(fs, "*.pt")))
        logging.info(f"[FEW-SHOT] tu task 1 tro di doc tu: {fs} ({n} file .pt)")
        if n == 0:
            raise FileNotFoundError(f"Thu muc few-shot rong: {fs}")
    else:
        logging.info("[FULL] moi task deu dung du lieu day du")

    # ── DataManager cho từng client ───────────────────────────────────────────
    logging.info(f"Tao DataManager cho {num_clients} client...")
    client_dms = []
    for c in range(num_clients):
        data_fl.set_client(c)
        dm = DataManager(args["dataset"], args["shuffle"], args["seed"],
                         args["init_cls"], args["increment"])
        # Ep dung kich thuoc task that: [6,6,6,6,5,5].
        # init_cls/increment cho ra [6,6,6,6,6,4] — sai voi bo du lieu nay.
        dm._increments = list(args.get("task_increments", data_fl.TASK_INCREMENTS))
        client_dms.append(dm)

    nb_tasks = client_dms[0].nb_tasks
    logging.info(f"Task increments: {client_dms[0]._increments} | {nb_tasks} task")

    global_model = factory.get_model(args["model_name"], args)
    local_models = [factory.get_model(args["model_name"], args) for _ in range(num_clients)]

    csv_path = os.path.join(run_dir, "metrics_round_by_round.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["task", "round_in_task", "global_round", "acc",
                     "prec_mic", "prec_mac", "prec_wei",
                     "rec_mic", "rec_mac", "rec_wei",
                     "f1_mic", "f1_mac", "f1_wei", "loss"])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_task, start_round, ckpt = 0, 0, None
    if args.get("resume"):
        path = args["resume"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Khong thay checkpoint: {path}")
        ckpt = torch.load(path, map_location=args["device"][0], weights_only=False)
        start_task = int(ckpt.get("task", 0))
        start_round = int(ckpt.get("round", 0))
        if start_round >= num_rounds:
            start_task, start_round = start_task + 1, 0
        logging.info(f"[RESUME] {os.path.basename(path)} -> Task {start_task}, "
                     f"Round {start_round}")

    for task in range(nb_tasks):
        # DER thêm một backbone mỗi task -> phải mở rộng kiến trúc cho MỌI task,
        # kể cả task bị bỏ qua khi resume.
        global_model.incremental_train(client_dms[0], skip_train=True)
        global_model.after_task()
        for c in range(num_clients):
            local_models[c].incremental_train(client_dms[c], skip_train=True)
            local_models[c].after_task()

        if task < start_task:
            continue

        if ckpt is not None and task == start_task:
            global_model._network.load_state_dict(ckpt["model_state_dict"])
            logging.info("[RESUME] Da nap trong so toan cuc.")

        known_before = sum(client_dms[0]._increments[:task])
        total_now = known_before + client_dms[0]._increments[task]
        global_model._network.to(args["device"][0])
        logging.info(f"\n===== TASK {task} | lop {known_before}-{total_now - 1} =====")

        r0 = start_round if task == start_task else 0
        for rnd in range(r0, num_rounds):
            ground = task * num_rounds + rnd
            logging.info(f"--- Task {task} | Round {rnd + 1}/{num_rounds} "
                         f"(Global {ground + 1}) ---")

            gstate = copy.deepcopy(global_model._network.state_dict())
            w_local, n_local, active = [], [], []

            for c in range(num_clients):
                dm = client_dms[c]
                n_new = len(dm.get_dataset(np.arange(known_before, total_now),
                                           source="train", mode="test"))
                if n_new == 0 and local_models[c]._get_memory() is None:
                    continue

                local_models[c]._network.load_state_dict(gstate)
                local_models[c]._network.to(args["device"][0])
                # Lui lai mot task de incremental_train tinh dung khoang lop,
                # va KHONG mo rong them backbone (da mo rong o tren).
                local_models[c]._cur_task = task - 1
                local_models[c]._known_classes = known_before
                args["start_round"] = 0
                local_models[c].incremental_train(dm)

                w_local.append({k: v.detach().cpu() for k, v
                                in local_models[c]._network.state_dict().items()})
                n_local.append(max(n_new, 1))
                active.append(c)

            if not w_local:
                logging.warning(f"Round {rnd + 1}: khong client nao tham gia — bo qua.")
                continue

            agg = average_weights(w_local,
                                  n_local if args.get("fedavg_weighted", True) else None)
            global_model._network.load_state_dict(agg)
            global_model._network.to(args["device"][0])
            logging.info(f"  FedAvg tu {len(active)}/{num_clients} client "
                         f"| tong mau: {sum(n_local):,}")

            # ── Đánh giá mô hình toàn cục ────────────────────────────────────
            global_model._cur_task = task
            global_model._known_classes = known_before
            global_model._total_classes = total_now
            global_model.test_loader = torch.utils.data.DataLoader(
                client_dms[0].get_dataset(np.arange(0, total_now),
                                          source="test", mode="test"),
                batch_size=args.get("eval_batch_size", 4096), shuffle=False,
                num_workers=0)
            m, _, _, _ = global_model.eval_task()
            logging.info(f"  [Task {task} | Round {rnd + 1}] acc {m['top1']:.2f}% | "
                         f"macro-F1 {m.get('f1_macro', 0):.2f}%")

            writer.writerow([task, rnd + 1, ground + 1, round(m["top1"], 4),
                             round(m.get("precision_micro", 0), 4),
                             round(m.get("precision_macro", 0), 4),
                             round(m.get("precision_weighted", 0), 4),
                             round(m.get("recall_micro", 0), 4),
                             round(m.get("recall_macro", 0), 4),
                             round(m.get("recall_weighted", 0), 4),
                             round(m.get("f1_micro", 0), 4),
                             round(m.get("f1_macro", 0), 4),
                             round(m.get("f1_weighted", 0), 4),
                             round(m.get("loss", 0), 6)])
            csv_file.flush()
            os.fsync(csv_file.fileno())

            _save_round_ckpt(ckpt_dir, global_model, task, rnd + 1, ground + 1, m,
                             known_before, args.get("keep_last_ckpt", 3))

        # ── Hết task: dựng exemplar memory rồi lưu checkpoint FINAL ───────────
        for c in range(num_clients):
            local_models[c]._network.load_state_dict(global_model._network.state_dict())
            local_models[c]._network.to(args["device"][0])
            local_models[c]._cur_task = task
            local_models[c]._known_classes = known_before
            local_models[c]._total_classes = total_now
            local_models[c].build_rehearsal_memory(client_dms[c],
                                                   local_models[c].samples_per_class)

        final = os.path.join(ckpt_dir, f"ckpt_task{task:02d}_FINAL.pth")
        torch.save({"task": task, "round": num_rounds,
                    "global_round": (task + 1) * num_rounds,
                    "model_state_dict": global_model._network.state_dict(),
                    "known_classes": total_now,
                    "fewshot_dir": fs or "full"}, final)
        logging.info(f"[CKPT-TASK] Da luu {os.path.basename(final)}")

        global_model._known_classes = total_now
        start_round = 0

    csv_file.close()
    logging.info(f"Xong. Metric: {csv_path}")


def _save_round_ckpt(ckpt_dir, model, task, rnd, ground, m, known, keep_last=3):
    name = f"ckpt_round{ground:04d}_task{task:02d}_r{rnd:03d}_acc{m['top1']:.1f}.pth"
    torch.save({"task": task, "round": rnd, "global_round": ground,
                "model_state_dict": model._network.state_dict(),
                "known_classes": known,
                "metrics": {"acc": m["top1"], "f1_mac": m.get("f1_macro", 0)}},
               os.path.join(ckpt_dir, name))
    # Don ban cu, GIU LAI moi file *_FINAL.pth
    files = sorted((os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                    if f.endswith(".pth") and not f.endswith("_FINAL.pth")),
                   key=os.path.getmtime)
    for old in files[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass
