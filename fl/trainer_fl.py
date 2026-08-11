"""SPCIL-FL — lớp federated (FedAvg) bọc quanh SPCIL, KHÔNG sửa một dòng nào của SPCIL.

Mô phỏng một tiến trình: mọi client chạy tuần tự trong cùng chương trình. Mỗi round
server nạp trọng số toàn cục vào từng client, client huấn luyện cục bộ, server lấy
trung bình có trọng số theo số mẫu — đúng FedAvg (McMahan et al. 2017). Dễ debug, dễ
lưu checkpoint và resume; cùng kiểu với AFSIC-IDS, HFIN, MalCL-FL trong dự án.

    python main_fl.py --config exps_fl/cic_iot23_fl.json

────────────────────────────────────────────────────────────────────────────────
BỐN CHỖ PHẢI TRÁNH KHI GỌI LẠI CODE CỦA SPCIL
────────────────────────────────────────────────────────────────────────────────
`DER.incremental_train` được viết cho bối cảnh TẬP TRUNG: một lần gọi = trọn một
task. Trong FL nó bị gọi 100 client × 30 round × 6 task, nên bốn hành vi sau trở
thành thảm hoạ nếu cứ gọi thẳng:

1. `incremental_train(skip_train=True)` VẪN dựng exemplar memory (der.py:72-78),
   trừ khi đặt `skip_rehearsal = True`. Vòng mở rộng kiến trúc sẽ chạy herding trên
   trọng số ngẫu nhiên cho 100 client × 6 task.
2. `incremental_train(...)` dựng lại exemplar memory sau MỖI lần huấn luyện
   (der.py:99). Trong FL thành 18.000 lượt herding thay vì 600.
3. `_init_train` đánh giá trên `test_loader` sau mỗi epoch nếu khác None
   (der.py:188). Tập test là 14 triệu mẫu — không được để client làm việc đó.
4. `_init_train` ghi một checkpoint sau MỖI epoch (der.py:199-209). 100 client cùng
   ghi đè một file, 18.000 lượt ghi vô nghĩa, lại trùng tên với checkpoint của FL.

Cách xử lý ở đây: KHÔNG gọi `incremental_train` để huấn luyện. Tự dựng train_loader
rồi gọi thẳng `model._train(train_loader, None)` — vẫn dùng nguyên `L_SP` + `L_RS`
của SPCIL — và chặn `torch.save` trong lúc đó. Exemplar memory chỉ dựng MỘT lần ở
cuối mỗi task.

────────────────────────────────────────────────────────────────────────────────
EXEMPLAR MEMORY PHẢI ĐI KÈM CHECKPOINT
────────────────────────────────────────────────────────────────────────────────
`_data_memory` / `_targets_memory` của từng client KHÔNG nằm trong
`model_state_dict` — chúng là mảng numpy riêng, chỉ sống trong RAM, và được nối
vào tập huấn luyện mỗi round qua `appendent=mdl._get_memory()`.

Trước 2026-08-11 checkpoint không mang chúng theo. Hậu quả: resume vào task ≥ 1
thì các task trước bị `continue` bỏ qua nên buffer không bao giờ được dựng,
`_get_memory()` trả về None, và client huấn luyện KHÔNG CÓ replay. Chương trình
vẫn chạy trơn, vẫn in ra số — chỉ là sai, và sai theo hướng làm phương pháp trông
tệ hơn thực tế.

Nay `ckpt_task*_FINAL.pth` mang theo `client_memory`, và resume vào task > 0 mà
không tìm được buffer thì DỪNG HẲN thay vì chạy tiếp. Checkpoint giữa task không
mang buffer (nó không đổi trong lúc task chạy) nên khi resume từ đó, code tự tìm
`ckpt_task{N-1:02d}_FINAL.pth` nằm cùng thư mục.

Dấu hiệu nhận biết trong log: dòng `tong mau:` đã tính cả appendent. Sang task 1
mà con số vẫn đúng bằng số mẫu lớp mới thì buffer đang rỗng.
"""
import contextlib
import copy
import csv
import glob
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import factory
from utils.data_manager import DataManager

from fl import data_fl


@contextlib.contextmanager
def _suppress_torch_save():
    """Chặn `torch.save` tạm thời — xem điểm 4 ở đầu file."""
    original = torch.save
    torch.save = lambda *a, **k: None
    try:
        yield
    finally:
        torch.save = original


def average_weights(w, weights=None):
    """FedAvg: theta = sum(n_k * theta_k) / sum(n_k). weights=None -> trung bình đều."""
    if weights is None:
        weights = [1.0] * len(w)
    tot = float(sum(weights))
    out = copy.deepcopy(w[0])
    for key in out.keys():
        ref = out[key]
        if not torch.is_floating_point(ref):
            acc = sum(float(w[i][key]) * weights[i] for i in range(len(w)))
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


def _thu_memory(local_models):
    """Gom exemplar buffer của mọi client thành dạng lưu được.

    Kích thước: num_clients x memory_size x feature_dim, float32. Với 100 client
    và memory_size 5000 trên bộ 33 chiều là khoảng 66 MB — chấp nhận được cho
    một file mỗi task.
    """
    return [(m._data_memory, m._targets_memory) for m in local_models]


def _nap_memory(local_models, buf):
    for m, (d, t) in zip(local_models, buf):
        m._data_memory, m._targets_memory = d, t


def _tim_memory(ckpt, path, start_task, num_clients):
    """Lấy exemplar buffer cho lần resume vào `start_task`.

    Thứ tự: buffer nằm sẵn trong file resume -> `ckpt_task{start_task-1}_FINAL.pth`
    cùng thư mục. Không thấy thì ném lỗi, KHÔNG chạy tiếp: thiếu buffer nghĩa là
    huấn luyện không replay, kết quả sai mà không có dấu hiệu gì.
    """
    buf, nguon = ckpt.get("client_memory"), os.path.basename(path)
    if buf is None:
        anh_em = os.path.join(os.path.dirname(path),
                              f"ckpt_task{start_task - 1:02d}_FINAL.pth")
        if not os.path.isfile(anh_em):
            raise FileNotFoundError(
                f"Resume vao task {start_task} nhung khong co exemplar memory.\n"
                f"  '{nguon}' khong chua khoa 'client_memory', va khong thay "
                f"'{os.path.basename(anh_em)}' trong '{os.path.dirname(path)}'.\n"
                f"  Chay tiep se huan luyen KHONG co replay -> ket qua sai.\n"
                f"  Checkpoint sinh truoc 2026-08-11 khong mang buffer; phai chay "
                f"lai tu task 0, hoac resume tu mot ckpt_task*_FINAL.pth moi.")
        buf = torch.load(anh_em, map_location="cpu",
                         weights_only=False).get("client_memory")
        nguon = os.path.basename(anh_em)
        if buf is None:
            raise KeyError(f"'{nguon}' khong chua 'client_memory' (checkpoint cu).")
    if len(buf) != num_clients:
        raise ValueError(f"'{nguon}' co buffer cua {len(buf)} client, "
                         f"lan chay nay can {num_clients}.")
    return buf, nguon


def _expand_one(model, dm, known_before):
    """Mở rộng kiến trúc DER thêm đúng một task, KHÔNG dựng exemplar memory.

    `skip_rehearsal = True` chặn herding trên trọng số ngẫu nhiên (điểm 1 ở đầu
    file).

    `known_before` PHẢI truyền vào, không được tin vào `model._known_classes`.
    `incremental_train` tính `_total_classes = _known_classes + task_size`, mà
    vòng huấn luyện đã ghi đè `_known_classes = known_before` của task ĐANG chạy
    (dòng ~311) và không khôi phục. Nên nếu để nguyên, sang task 1 client sẽ tính
    `0 + 6 = 6` thay vì `12`: `fc` của client thành (6,128) còn global là (12,128),
    và `load_state_dict` ở dòng ~321 nổ ngay round đầu của task 1.

    Bản trước 2026-08-11 mắc đúng lỗi này. Nó không lộ ra vì lần chạy thật mới chỉ
    đi hết task 0.
    """
    model._known_classes = known_before
    model.skip_rehearsal = True
    model.incremental_train(dm, skip_train=True)
    model.after_task()
    model.skip_rehearsal = False


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
        # `ckpt_task` = task mà TRỌNG SỐ trong file thuộc về; `start_task` = task
        # sẽ huấn luyện tiếp. Hai giá trị này khác nhau khi file là FINAL của một
        # task đã xong, và phải giữ riêng — xem chỗ nạp trọng số trong vòng task.
        ckpt_task = int(ckpt.get("task", 0))
        start_task = ckpt_task
        start_round = int(ckpt.get("round", 0))
        if start_round >= num_rounds:
            start_task, start_round = start_task + 1, 0
        logging.info(f"[RESUME] {os.path.basename(path)} -> Task {start_task}, "
                     f"Round {start_round}")

        # Task 0 chua co exemplar nao nen buffer rong la dung; tu task 1 tro di
        # thi buffer la bat buoc (xem khoi EXEMPLAR MEMORY o dau file).
        if start_task > 0:
            buf, nguon = _tim_memory(ckpt, path, start_task, num_clients)
            _nap_memory(local_models, buf)
            logging.info(f"[RESUME] Da nap exemplar memory tu {nguon} | "
                         f"client 0: {len(buf[0][1]):,} mau")

    bs = args["batch_size"]
    nw = args.get("num_workers", 0)

    for task in range(nb_tasks):
        known_before = sum(client_dms[0]._increments[:task])
        total_now = known_before + client_dms[0]._increments[task]

        # Mở rộng kiến trúc thêm đúng một task, KHÔNG dựng exemplar (điểm 1).
        # Phải làm cho MỌI task kể cả task bị bỏ qua khi resume, vì DER thêm
        # một backbone mỗi task.
        _expand_one(global_model, client_dms[0], known_before)
        for c in range(num_clients):
            _expand_one(local_models[c], client_dms[c], known_before)

        # Nạp trọng số NGAY SAU khi kiến trúc vừa đạt đúng hình dạng của checkpoint,
        # tức ở lượt `ckpt_task` chứ không phải `start_task`. DER thêm một backbone
        # mỗi task, nên FINAL của task N có N+1 backbone; đợi đến lượt task N+1 mới
        # nạp thì kiến trúc đã phình thêm một backbone và `load_state_dict` nổ.
        # Với checkpoint giữa task thì ckpt_task == start_task, nhánh này trùng chỗ cũ.
        if ckpt is not None and task == ckpt_task:
            global_model._network.load_state_dict(ckpt["model_state_dict"])
            logging.info(f"[RESUME] Da nap trong so toan cuc (kien truc task {task}).")

        if task < start_task:
            continue

        global_model._network.to(args["device"][0])
        logging.info(f"\n===== TASK {task} | lop {known_before}-{total_now - 1} | "
                     f"{len(global_model._network.convnets)} backbone =====")

        r0 = start_round if task == start_task else 0
        for rnd in range(r0, num_rounds):
            ground = task * num_rounds + rnd
            logging.info(f"--- Task {task} | Round {rnd + 1}/{num_rounds} "
                         f"(Global {ground + 1}) ---")

            gstate = copy.deepcopy(global_model._network.state_dict())
            w_local, n_local, active = [], [], []

            for c in range(num_clients):
                dm = client_dms[c]
                mdl = local_models[c]

                mdl._cur_task = task
                mdl._known_classes = known_before
                mdl._total_classes = total_now

                train_set = dm.get_dataset(np.arange(known_before, total_now),
                                           source="train", mode="train",
                                           appendent=mdl._get_memory())
                n_new = len(train_set)
                if n_new == 0:
                    continue

                mdl._network.load_state_dict(gstate)
                mdl._network.to(args["device"][0])
                # DER: đóng băng các backbone của task trước, y như der.py:43-46
                if task > 0:
                    for i in range(task):
                        for p in mdl._network.convnets[i].parameters():
                            p.requires_grad = False

                loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                                    num_workers=nw)
                args["start_round"] = 0
                # test_loader = None -> client KHONG danh gia tren 14 trieu mau
                # (diem 3); chan torch.save de khong ghi checkpoint moi epoch (diem 4)
                with _suppress_torch_save():
                    mdl._train(loader, None)

                w_local.append({k: v.detach().cpu() for k, v
                                in mdl._network.state_dict().items()})
                n_local.append(n_new)
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
            # Chi client TEST_OWNER co tap test (xem data_fl.download_data)
            global_model.test_loader = DataLoader(
                client_dms[data_fl.TEST_OWNER].get_dataset(
                    np.arange(0, total_now), source="test", mode="test"),
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

        # ── Hết task: dựng exemplar memory MỘT LẦN cho mỗi client ─────────────
        logging.info(f"  Dung exemplar memory cho {num_clients} client...")
        for c in range(num_clients):
            mdl = local_models[c]
            mdl._network.load_state_dict(global_model._network.state_dict())
            mdl._network.to(args["device"][0])
            mdl._cur_task = task
            mdl._known_classes = known_before
            mdl._total_classes = total_now
            with _suppress_torch_save():
                mdl.build_rehearsal_memory(client_dms[c], mdl.samples_per_class)

        final = os.path.join(ckpt_dir, f"ckpt_task{task:02d}_FINAL.pth")
        torch.save({"task": task, "round": num_rounds,
                    "global_round": (task + 1) * num_rounds,
                    "model_state_dict": global_model._network.state_dict(),
                    "known_classes": total_now,
                    "fewshot_dir": fs or "full",
                    # Bat buoc de resume vao task sau con replay - xem dau file.
                    "client_memory": _thu_memory(local_models)}, final)
        logging.info(f"[CKPT-TASK] Da luu {os.path.basename(final)} | "
                     f"exemplar client 0: {local_models[0].exemplar_size:,} | "
                     f"{os.path.getsize(final) / 1048576:.1f} MB")

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
