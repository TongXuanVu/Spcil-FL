"""Bộ nạp dữ liệu FEDERATED cho SPCIL-FL — CIC-IoT23 và CAN-bus/IoV, 100 client.

Đọc `client_<cid>_task_<t>.pt` của ĐÚNG MỘT client. Trainer tạo một DataManager cho
mỗi client, mỗi lần gọi `set_client(cid)` trước.

Ba kịch bản, chọn bằng `set_fewshot_dir()`:

    None                        -> full data, task 1..NUM_TASKS
    .../federated_data_fewshot  -> 1%,      task 2..NUM_TASKS (task 1 vẫn lấy full)
    .../federated_data_10shot   -> 10-shot, task 2..NUM_TASKS (task 1 vẫn lấy full)

Hai thư mục few-shot cố tình không có task 1, vì task 0 (base) dùng chung cho cả ba
kịch bản — đúng quy ước của AFSIC-IDS và HFIN (IoV cũng theo quy ước này, xem
AFSIC-IOV/tools/make_fewshot_iov100.py: TASK_BASE = 1).

Hỗ trợ nhiều bộ dữ liệu qua CAU_HINH_BO bên dưới. `register()` tự nhận diện bộ
dữ liệu theo tên trong `--dataset`/config JSON và gọi `set_dataset()` hộ — không
cần sửa main_fl.py/trainer_fl.py.

LƯU Ý — `class_order` luôn là 0..N-1 bất kể client đó giữ lớp nào (N = số lớp của
bộ dữ liệu đang chọn). Nếu để DataManager tự suy từ `np.unique(train_targets)` thì
mỗi client sẽ có thứ tự lớp khác nhau, nhãn bị ánh xạ lệch, và mô hình gộp lại vô
nghĩa. Vì đây là identity map [0,1,...,N-1] và nhãn IoT/IoV đều đã tuần tự sẵn
(IoV không cần remap — xem CLAUDE.md), giữ class_order dài hơn N một chút KHÔNG
gây lỗi (order.index(label) vẫn đúng); nhưng vẫn cập nhật đúng N trong set_dataset()
để n_lop hiển thị/log chính xác.
"""
import glob
import os

import numpy as np
import torch

# ── Cấu hình theo từng bộ dữ liệu ───────────────────────────────────────────────
# increments: kích thước từng task, khớp cách chia thật của bộ 100 client — KHÔNG
# suy từ init_cls/increment (công thức đó cho ra sai số, vd IoT sẽ ra [6,6,6,6,6,4]).
CAU_HINH_BO = {
    "cic_iot23": dict(
        num_classes=34,
        num_tasks=6,
        increments=[6, 6, 6, 6, 5, 5],
        names=("cic_iot23_fl", "ciciot23_fl", "cic-iot23-fl"),
        n_feat_fallback=33,
    ),
    "can_iov": dict(
        num_classes=13,
        num_tasks=5,
        increments=[3, 3, 3, 2, 2],
        names=("can_iov_fl", "caniov_fl", "can-iov-fl", "iov_fl", "iov100_fl"),
        n_feat_fallback=31,
    ),
}
_ALL_NAMES = {ten: key for key, cfg in CAU_HINH_BO.items() for ten in cfg["names"]}

BO_HIEN_TAI = "cic_iot23"                              # mặc định, giữ tương thích ngược
NUM_CLASSES = CAU_HINH_BO[BO_HIEN_TAI]["num_classes"]
NUM_TASKS = CAU_HINH_BO[BO_HIEN_TAI]["num_tasks"]
TASK_INCREMENTS = list(CAU_HINH_BO[BO_HIEN_TAI]["increments"])
DATASET_NAMES = CAU_HINH_BO[BO_HIEN_TAI]["names"]

# Chỉ client này nạp tập test — xem ghi chú trong download_data().
# Trainer đánh giá mô hình toàn cục bằng client_dms[TEST_OWNER].
# Đúng cho cả hai bộ: client 0 luôn nằm trong dải 0-49 có mặt ở mọi task.
TEST_OWNER = 0

# ── Trạng thái toàn cục, trainer đặt trước khi tạo mỗi DataManager ─────────────
CLIENT_ID = 0
FEWSHOT_DIR = None
DATA_ROOT = None


def set_dataset(ten):
    """Chuyển cấu hình toàn cục (NUM_CLASSES/NUM_TASKS/TASK_INCREMENTS/class_order)
    sang bộ dữ liệu `ten` (khoá trong CAU_HINH_BO, vd "can_iov"). `register()` gọi
    hàm này tự động khi nhận ra tên dataset trong config; cũng có thể gọi tay từ
    ngoài trước khi train nếu muốn log tường minh.
    """
    global BO_HIEN_TAI, NUM_CLASSES, NUM_TASKS, TASK_INCREMENTS, DATASET_NAMES
    if ten not in CAU_HINH_BO:
        raise SystemExit(f"[FL-DATA] Dataset khong ho tro: {ten}. Chi co: {list(CAU_HINH_BO)}")
    cfg = CAU_HINH_BO[ten]
    BO_HIEN_TAI = ten
    NUM_CLASSES = cfg["num_classes"]
    NUM_TASKS = cfg["num_tasks"]
    TASK_INCREMENTS = list(cfg["increments"])
    DATASET_NAMES = cfg["names"]
    # class_order la class attribute (khong phai bien module) -> phai gan truc tiep
    # tren class de moi instance moi tao sau nay doc dung, khong the "reassign" ten
    # module-level vi khong ai import class_order rieng ca.
    iCICIoT23FL.class_order = list(range(NUM_CLASSES))
    # Xoa cache test/n_feat cu (phong truong hop doi bo dataset trong cung process,
    # vd chay thu nghiem nhieu bo lien tiep khong khoi dong lai Python).
    if hasattr(iCICIoT23FL, "_test_cache"):
        del iCICIoT23FL._test_cache
    if hasattr(iCICIoT23FL, "_n_feat"):
        del iCICIoT23FL._n_feat
    print(f"[FL-DATA] set_dataset({ten}): {NUM_CLASSES} lop, {NUM_TASKS} task, "
          f"increments={TASK_INCREMENTS}")


def set_client(cid):
    global CLIENT_ID
    CLIENT_ID = int(cid)


def set_fewshot_dir(path):
    global FEWSHOT_DIR
    FEWSHOT_DIR = path or None


def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = path or None


# Các vị trí thường gặp của bộ 100 client, thử theo thứ tự. Ưu tiên `--data_root`.
_LOCAL_CANDIDATES = [
    r"D:\FL\core\data_split\100 client",          # o D (sau khi chuyen tu C)
    r"C:\FederatedLearning\FL\core\data_split\100 client",
    "/mnt/d/FL/core/data_split/100 client",       # WSL
]


def _auto_root():
    """Tìm thư mục chứa client_*_task_*.pt.

    Thứ tự: --data_root -> /kaggle/input -> các đường dẫn quen thuộc -> thư mục
    anh em của repo. Mỗi ứng viên đều thử cả `<dir>/federated_data` lẫn `<dir>`
    vì hai bố cục đều tồn tại trong dự án.
    """
    def _ok(d):
        for sub in (os.path.join(d, "federated_data"), d):
            if os.path.exists(os.path.join(sub, "client_0_task_1.pt")):
                return sub
        return None

    if DATA_ROOT:
        hit = _ok(DATA_ROOT)
        if hit:
            return hit
        raise FileNotFoundError(
            f"--data_root tro toi '{DATA_ROOT}' nhung khong thay client_0_task_1.pt "
            f"o do hay trong '{os.path.join(DATA_ROOT, 'federated_data')}'")

    if os.path.exists("/kaggle/input"):
        hits = glob.glob("/kaggle/input/**/client_0_task_1.pt", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
        raise FileNotFoundError("Khong tim thay client_0_task_1.pt trong /kaggle/input")

    tried = []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in _LOCAL_CANDIDATES + [
            os.path.join(os.path.dirname(here), "FL", "core", "data_split", "100 client")]:
        tried.append(cand)
        hit = _ok(cand)
        if hit:
            return hit

    raise FileNotFoundError(
        "Khong tim thay du lieu 100 client. Da thu:\n  " + "\n  ".join(tried)
        + "\nDung --data_root de chi dinh truc tiep, vi du:\n"
        + r'  --data_root "D:\FL\core\data_split\100 client"')


def _find_test_file(root):
    """Tìm global_test_data.pt: cùng thư mục, thư mục cha, rồi thư mục ông."""
    up1 = os.path.dirname(root)
    up2 = os.path.dirname(up1)
    for p in (os.path.join(root, "global_test_data.pt"),
              os.path.join(up1, "global_test_data.pt"),
              os.path.join(up2, "global_test_data.pt")):
        if os.path.exists(p):
            return p
    for base in (up1, up2):
        hits = glob.glob(os.path.join(base, "**", "global_test_data.pt"), recursive=True)
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

        n_feat_fb = CAU_HINH_BO[BO_HIEN_TAI]["n_feat_fallback"]
        if xs:
            self.train_data = np.concatenate(xs).astype(np.float32)
            self.train_targets = np.concatenate(ys).astype(np.int64)
        else:
            self.train_data = np.zeros((0, n_feat_fb), dtype=np.float32)
            self.train_targets = np.zeros((0,), dtype=np.int64)

        # ── Tập test: CHỈ client 0 nạp thật ──────────────────────────────────
        #
        # Trainer chỉ dùng `client_dms[0]` để đánh giá mô hình toàn cục; 99 client
        # còn lại không bao giờ đụng tới tập test. Nếu client nào cũng nạp thì:
        #
        #   - `DataManager._setup_data` gọi `_map_new_class_index` trên 14 triệu
        #     nhãn, mà hàm đó là vòng lặp Python `order.index(x)` — khoảng 476
        #     triệu phép tra cứu, mất ~13 giây MỖI client (22 phút cho 100 client);
        #   - mỗi client giữ riêng một mảng int64 14 triệu phần tử = 0,11 GB,
        #     tổng 11,2 GB RAM vứt đi.
        #
        # Client khác nhận mảng rỗng đúng số chiều — `get_dataset(source="test")`
        # của chúng trả về tập rỗng, không ai gọi nên vô hại.
        if cid == TEST_OWNER:
            if not hasattr(iCICIoT23FL, "_test_cache"):
                tx, ty = _load(_find_test_file(root))
                iCICIoT23FL._test_cache = (
                    (tx.numpy() if torch.is_tensor(tx) else np.asarray(tx)).astype(np.float32),
                    (ty.numpy() if torch.is_tensor(ty) else np.asarray(ty)).astype(np.int64),
                )
                iCICIoT23FL._n_feat = iCICIoT23FL._test_cache[0].shape[1]
            self.test_data, self.test_targets = iCICIoT23FL._test_cache
        else:
            n_feat = getattr(iCICIoT23FL, "_n_feat", None) \
                or (self.train_data.shape[1] if len(self.train_data) else n_feat_fb)
            self.test_data = np.zeros((0, n_feat), dtype=np.float32)
            self.test_targets = np.zeros((0,), dtype=np.int64)

        n_feat = self.train_data.shape[1] if len(self.train_data) else self.test_data.shape[1]
        tag = " | co tap test" if cid == TEST_OWNER else ""
        print(f"[FL-DATA] Client {cid:>3}: train={len(self.train_targets):>9,} mau | "
              f"{len(np.unique(self.train_targets)):>2} lop | {n_feat} dac trung"
              + (f" | few-shot: {os.path.basename(FEWSHOT_DIR)}" if FEWSHOT_DIR else "")
              + tag)


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
        ten = str(dataset_name).lower()
        key = _ALL_NAMES.get(ten)
        if key:
            if key != BO_HIEN_TAI:
                set_dataset(key)
            return iCICIoT23FL()
        return original(dataset_name)

    dm._get_idata = patched
    dm._fl_registered = True
    print(f"[FL-DATA] Da dang ky dataset: {', '.join(_ALL_NAMES)}")
