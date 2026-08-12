# SPCIL-FL — kịch bản FULL, checkpoint giữa chừng (global round 155/180)

Phiên `11-08-26 13-26`, resume từ `../ckpt_task00_full.pth`. Xong task 1–4,
đang ở task 5 round 5/30.

    ckpt_round0155_task05_r005_acc73.5.pth   trong so, kien truc task 5
    ckpt_task04_FINAL.pth                    exemplar memory (client 0: 4.685 mau)

**Hai file phải nằm CÙNG thư mục** — checkpoint giữa task không mang buffer,
`_tim_memory` tìm `ckpt_task04_FINAL.pth` ngay bên cạnh file resume.

    python main_fl.py --config exps_fl/cic_iot23_fl.json \
        --data_root "<.../100client>" \
        --resume resume_checkpoints/spcil_full/ckpt_round0155_task05_r005_acc73.5.pth

macro-F1 round cuối mỗi task: 40.00 / 14.15 / 22.07 / 17.21 / 26.16 / 37.49 (r5).
