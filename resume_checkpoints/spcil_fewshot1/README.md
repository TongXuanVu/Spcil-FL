# SPCIL-FL — kịch bản 1%, checkpoint giữa chừng (global round 109/180)

Lần chạy `11-08-26 09-01`, resume từ `../ckpt_task00_full.pth`. Đã xong task 1 và
task 2, đang ở task 3 round 19/30.

    ckpt_round0109_task03_r019_acc77.6.pth   trong so, kien truc task 3
    ckpt_task02_FINAL.pth                    exemplar memory cua 100 client

**Hai file phải nằm CÙNG một thư mục.** Checkpoint giữa task không mang buffer
(buffer không đổi trong lúc một task chạy), nên `_tim_memory` trong `trainer_fl.py`
tự tìm `ckpt_task{N-1}_FINAL.pth` bên cạnh file resume. Tách hai file ra hai chỗ
là chương trình dừng với thông báo thiếu memory.

    python main_fl.py --config exps_fl/cic_iot23_fl_fewshot1.json \
        --data_root "<.../100client>" \
        --fewshot_dir "<.../federated_data_fewshot>" \
        --resume resume_checkpoints/spcil_fewshot1/ckpt_round0109_task03_r019_acc77.6.pth

Kết quả tới lúc này (macro-F1 round cuối mỗi task): task 1 = 33.11, task 2 = 16.22,
task 3 ở round 19 = 26.64.
