# Checkpoint task 0 dùng chung cho cả ba kịch bản

`ckpt_task00_full.pth` — mô hình toàn cục sau 30 round của task 0, lần chạy
`09-08-26 12:09` (100 client, seed 42, full data). acc 99.45 | macro-F1 40.00.

**Cả ba kịch bản đều resume được từ file này.** Task 0 luôn huấn luyện trên full
data ở mọi kịch bản: `fl/data_fl.py` chọn thư mục bằng
`base = FEWSHOT_DIR if (FEWSHOT_DIR and t > 1) else root`, nên file `client_*_task_1.pt`
(tức task 0) luôn đọc từ thư mục full. Đã kiểm: dữ liệu task 0 trùng khít từng
byte giữa bản full và bản few-shot.

File này **không chứa** `client_memory` vì sinh trước 2026-08-11. Không sao —
`trainer_fl.py` nhận ra nó thuộc task 0 và tự dựng lại buffer từ chính trọng số
này. Herding chỉ dùng `argmin`, không có RNG, và trọng số ở đây trùng khít với
trọng số mà `ckpt_task00_FINAL.pth` lẽ ra chứa, nên buffer dựng lại là chính xác
chứ không phải xấp xỉ.

Lần chạy 09-08 chết đúng ở bước dựng memory sau round 30, nên không có
`ckpt_task00_FINAL.pth`.

    python main_fl.py --config exps_fl/cic_iot23_fl.json \
        --data_root "<.../100client>" \
        --resume resume_checkpoints/ckpt_task00_full.pth

`metrics_task00_full.csv` là 30 dòng metric của task 0, để ghép vào bảng tổng hợp.
