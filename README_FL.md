# SPCIL-FL

Bản federated (FedAvg) của SPCIL cho CIC-IoT23, 100 client. **Không sửa một dòng nào
của SPCIL gốc** — toàn bộ phần FL nằm trong thư mục `fl/`, gắn vào SPCIL bằng
monkey-patch lúc chạy.

---

## 1. Cài đặt

Repo này chỉ chứa **lớp FL**. Code SPCIL lấy từ repo gốc:

```bash
git clone https://github.com/TongXuanVu/Spcil-fl.git
cd Spcil-fl

# Keo code SPCIL goc vao cung thu muc (khong ghi de file nao cua repo nay)
git clone --depth 1 <URL-repo-SPCIL> _spcil_tmp
robocopy _spcil_tmp . /E /XD .git /XF README.md .gitignore
rmdir /s /q _spcil_tmp
```

Sau bước này thư mục phải có cả hai nhóm:

```
main.py  trainer.py  models/  utils/  convs/  losses/  exps/     <- SPCIL goc
main_fl.py  fl/  exps_fl/  README_FL.md                          <- lop FL
```

`python main.py --config exps/cic_iot23_der.json` vẫn chạy bản tập trung như cũ.

---

## 2. Thiết kế

| File | Vai trò |
|---|---|
| `fl/data_fl.py` | Nạp `client_<cid>_task_<t>.pt` của đúng một client; `register()` gắn dataset `cic_iot23_fl` vào `DataManager` bằng monkey-patch |
| `fl/trainer_fl.py` | Vòng lặp task / round / client + FedAvg + checkpoint/resume |
| `main_fl.py` | Điểm vào |
| `exps_fl/*.json` | Ba config cho ba kịch bản |

**FedAvg**: `θ = Σ nₖ·θₖ / Σ nₖ` (McMahan et al. 2017). Mô phỏng một tiến trình —
mọi client chạy tuần tự trong cùng chương trình Python. Cùng kiểu hiện thực hoá với
AFSIC-IDS, HFIN, MalCL-FL. Đặt `"fedavg_weighted": false` nếu muốn trung bình đều.

**`models/der.py` giữ nguyên**, kể cả hai loss riêng của SPCIL là `L_SP` (AdaSP) và
`L_RS` (Multi-Similarity).

### Ba chỗ dễ sai đã xử lý sẵn

1. **Thứ tự lớp.** `DataManager` của SPCIL tự suy thứ tự từ `np.unique(train_targets)`.
   Client chỉ giữ vài lớp thì mỗi client ra một thứ tự khác nhau, nhãn ánh xạ lệch,
   mô hình gộp lại vô nghĩa. `iCICIoT23FL.class_order` cố định `0..33`.
2. **Kích thước task.** `init_cls=6, increment=6` cho `[6,6,6,6,6,4]`, nhưng bộ dữ
   liệu chia thật là `[6,6,6,6,5,5]`. Trainer ghi đè `dm._increments`.
3. **Tập test 1,96 GB** cache ở cấp lớp — 100 DataManager dùng chung, nếu không sẽ
   tốn 196 GB RAM.

---

## 3. Chạy

```python
ROOT = "/kaggle/input/datasets/tongxuanvu/iot100client"
DATA = f"{ROOT}/100client"
FS1  = f"{ROOT}/iot100client_fewshot/federated_data_fewshot"
NS   = f"{ROOT}/iot100client_fewshot/federated_data_10shot"
```

Chạy **1% trước** để có sẵn `ckpt_task00_FINAL.pth` cho 10-shot dùng lại — task 0 của
cả ba kịch bản đều là full data.

```bash
# 1%
python main_fl.py --config exps_fl/cic_iot23_fl_fewshot1.json \
    --data_root "$DATA" --fewshot_dir "$FS1"

# 10-shot, resume task 0 tu lan chay 1%
python main_fl.py --config exps_fl/cic_iot23_fl_10shot.json \
    --data_root "$DATA" --fewshot_dir "$NS" \
    --resume logs/spcil_fl/cic_iot23_fl/<run-1%>/checkpoints/ckpt_task00_FINAL.pth

# full data
python main_fl.py --config exps_fl/cic_iot23_fl.json --data_root "$DATA"
```

---

## 4. ⚠️ Chạy thử trước — code này CHƯA chạy lần nào

```bash
python main_fl.py --config exps_fl/cic_iot23_fl.json --data_root "$DATA" --debug
```

(5 client, 2 round, 1 epoch — vài phút). Phải thấy đủ chuỗi:

```
[FL-DATA] Da dang ky dataset: cic_iot23_fl, ciciot23_fl, cic-iot23-fl
[FL-DATA] Client   0: train=... mau | .. lop | 33 dac trung
Task increments: [6, 6, 6, 6, 5, 5] | 6 task
===== TASK 0 | lop 0-5 =====
  FedAvg tu N/5 client | tong mau: ...
  [Task 0 | Round 1] acc ..% | macro-F1 ..%
[CKPT-TASK] Da luu ckpt_task00_FINAL.pth
```

**Chỗ rủi ro nhất: DER thêm một backbone mỗi task**, mà `incremental_train` bị gọi
nhiều lần (một lần dựng kiến trúc + mỗi round một lần huấn luyện). Trainer đặt lại
`_cur_task = task - 1` trước mỗi lần gọi huấn luyện để tránh cộng dồn. Trong lần
debug **phải kiểm số tham số**: nó chỉ được tăng khi sang task mới, không tăng theo
từng round. Nếu phình theo round thì `_cur_task` chưa được lùi đúng chỗ.

---

## 5. Cần thấy gì trong log khi chạy thật

1. `[FL-DATA] Client 0: ... | 33 dac trung` — đúng CIC-IoT23, không phải IoV 31
2. `Task increments: [6, 6, 6, 6, 5, 5] | 6 task`
3. `[FEW-SHOT] tu task 1 tro di doc tu: .../federated_data_fewshot (383 file .pt)` —
   không có dòng này nghĩa là đang chạy full data
4. Số client tham gia mỗi round tăng dần theo task (bộ dữ liệu chia kiểu client vào
   dần), giống HFIN
5. `[CKPT-TASK] Da luu ckpt_taskNN_FINAL.pth` sau mỗi task — file `*_FINAL.pth`
   **không bao giờ bị dọn**; checkpoint theo round chỉ giữ 3 bản mới nhất

## 6. Kết quả

Thêm vào `Tổng hợp kết quả/iot100/` theo schema của `hfin_der_180_rounds.csv`:

```
spcil_fl_180_rounds.csv
spcil_fl_fewshot1_180_rounds.csv
spcil_fl_10shot_180_rounds.csv
```

Bản SPCIL **tập trung** giữ nhãn riêng "tập trung, cận trên" — không xếp ngang hàng
với các phương pháp federated.
