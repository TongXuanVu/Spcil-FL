"""Bộ nạp dữ liệu FEDERATED cho SPCIL-FL — CIC-IoT23, 100 client.

Đọc `client_<cid>_task_<t>.pt` của ĐÚNG MỘT client. Trainer tạo một DataManager cho
mỗi client, mỗi lần gọi `set_client(cid)` trước.

Ba kịch bản, chọn bằng `set_fewshot_dir()`:

    None                        -> full data, task 1..6
    .../federated_data_fewshot  -> 1%,      task 2..6 (task 1 vẫn lấy full)
    .../federated_data_10shot   -> 10-shot, task 2..6 (task 1 vẫn lấy full)

Hai thư mục few-shot cố tình không có task 1, vì task 0 (base) dùng chung cho cả ba
kịch bản — đúng quy ước của AFSIC-IDS và HFIN.

LƯU Ý — `class_order` luôn là 0..33 bất kể client đó giữ lớp nào. Nếu để DataManager
tự suy từ `np.unique(train_targets)` thì mỗi client sẽ có thứ tự lớp khác nhau, nhãn
bị ánh xạ lệch, và mô hình gộp lại vô nghĩa.
"""
import glob
import os

import numpy as np
import torch

NUM_CLASSES = 34
NUM_TASKS = 6

# Kích thước từng task, khớp với cách chia thật của bộ 100 client.
# KHÔNG suy từ init_cls/increment: công thức đó cho [6,6,6,6,6,4].
TASK_INCREMENTS = [6, 6, 6, 6, 5, 5]

DATASET_NAMES = ("cic_iot23_fl", "ciciot23_fl", "cic-iot23-fl")

# ── Trạng thái toàn cục, trainer đặt trước khi tạo mỗi DataManager ─────────────
CLIENT_ID = 0
FEWSHOT_DIR = None
DATA_ROOT = None


def set_client(cid):
    global CLIENT_ID
    CLIENT_ID = int(cid)


def set_fewshot_dir(path):
    global FEWSHOT_DIR
    FEWSHOT_DIR = path or None


def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = path or None


def _auto_root():
    if DATA_ROOT:
        return DATA_ROOT
    if os.path.exists("/kaggle/input"):
        hits = glob.glob("/kaggle/input/**/client_0_task_1.pt", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
        raise FileNotFoundError("Khong tim thay client_0_task_1.pt trong /kaggle/input")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(os.path.dirname(here), "FL", "core", "data_split", "100 client")
    for sub in (os.path.join(cand, "federated_data"), cand):
        if os.path.exists(os.path.join(sub, "client_0_task_1.pt")):
            return sub
    raise FileNotFoundError(f"Khong tim thay du lieu 100 client (da thu {cand})")


def _find_test_file(root):
    for p in (os.path.join(root, "global_test_data.pt"),
              os.path.join(os.path.dirname(root), "global_test_data.pt")):
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(os.path.dirname(root), "**", "global_test_data.pt"),
                     recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Khong tim thay global_test_data.pt quanh {root}")


def _load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        return d["x"], d["y"]
    return d[0], d[1]


class iCICIoT23FL:
    """Adapter tương thích DataManager của SPCIL, chỉ nạp dữ liệu 1 client."""

    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = list(range(NUM_CLASSES))     # cố định cho MỌI client

    def download_data(self):
        root = _auto_root()
        cid = CLIENT_ID

        xs, ys = [], []
        for t in range(1, NUM_TASKS + 1):
            # Few-shot chỉ có task 2..6; task 1 luôn lấy từ thư mục full.
            base = FEWSHOT_DIR if (FEWSHOT_DIR and t > 1) else root
            path = os.path.join(base, f"client_{cid}_task_{t}.pt")
            if not os.path.exists(path):
                continue
            x, y = _load(path)
            if len(y) == 0:
                continue
            xs.append(x.numpy() if torch.is_tensor(x) else np.asarray(x))
            ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))

        if xs:
            self.train_data = np.concatenate(xs).astype(np.float32)
            self.train_targets = np.concatenate(ys).astype(np.int64)
        else:
            self.train_data = np.zeros((0, 33), dtype=np.float32)
            self.train_targets = np.zeros((0,), dtype=np.int64)

        # Tập test dùng chung — cache ở cấp lớp để 100 DataManager không giữ
        # 100 bản sao của file 1,96 GB.
        if not hasattr(iCICIoT23FL, "_test_cache"):
            tx, ty = _load(_find_test_file(root))
            iCICIoT23FL._test_cache = (
                (tx.numpy() if torch.is_tensor(tx) else np.asarray(tx)).astype(np.float32),
                (ty.numpy() if torch.is_tensor(ty) else np.asarray(ty)).astype(np.int64),
            )
        self.test_data, self.test_targets = iCICIoT23FL._test_cache

        n_feat = self.train_data.shape[1] if len(self.train_data) else self.test_data.shape[1]
        print(f"[FL-DATA] Client {cid:>3}: train={len(self.train_targets):>9,} mau | "
              f"{len(np.unique(self.train_targets)):>2} lop | {n_feat} dac trung"
              + (f" | few-shot: {os.path.basename(FEWSHOT_DIR)}" if FEWSHOT_DIR else ""))


def register():
    """Gắn dataset FL vào DataManager của SPCIL mà KHÔNG sửa file nào của SPCIL.

    `utils.data_manager._get_idata` là một hàm module-level; ở đây bọc nó lại: tên
    dataset FL thì trả adapter của mình, còn lại chuyển tiếp cho bản gốc. Nhờ vậy
    SPCIL giữ nguyên 100% và có thể clone thẳng từ repo gốc.
    """
    from utils import data_manager as dm

    if getattr(dm, "_fl_registered", False):
        return
    original = dm._get_idata

    def patched(dataset_name):
        if str(dataset_name).lower() in DATASET_NAMES:
            return iCICIoT23FL()
        return original(dataset_name)

    dm._get_idata = patched
    dm._fl_registered = True
    print(f"[FL-DATA] Da dang ky dataset: {', '.join(DATASET_NAMES)}")
