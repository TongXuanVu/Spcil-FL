# SPCIL-FL — kịch bản 1%, ĐÃ CHẠY XONG 180/180 round

Hai phiên Kaggle, cùng resume từ `../ckpt_task00_full.pth`:

| phiên | global round | ghi chú |
|---|---|---|
| `11-08-26 09-01` | 31–109 | xong task 1, 2; đứt giữa task 3 |
| `11-08-26 13-26` | 110–180 | chạy nốt task 3, 4, 5 |

`metrics_spcil_fl_fewshot1_180_rounds.csv` là bản gộp của cả hai, 150 dòng liền
mạch từ global_round 31 đến 180, không thiếu round nào. Task 0 (round 1–30)
không có ở đây vì dùng chung với bản full — lấy từ `../metrics_task00_full.csv`.

## Kết quả (macro-F1, round cuối mỗi task)

| task | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1% | 40.00* | 33.11 | 16.22 | 27.49 | 16.54 | 27.91 |

\* task 0 dùng chung với bản full.

Không sụp đổ theo task — dao động 16–33 chứ không rơi về 0 như F2SCIL hay
AFSIC-IoV. Điểm yếu của SPCIL trong FL nằm ở mức tuyệt đối (task 0 chỉ đạt 40.00
so với 68.83 của bản tập trung, và quá nửa lượt huấn luyện cục bộ có
`Train_accy 0.00`), không phải ở khả năng giữ lớp cũ.

## Checkpoint

`ckpt_task05_FINAL.pth` — mô hình cuối cùng, kèm `client_memory` của 100 client
(exemplar client 0: 2.034 mẫu). Đủ để đánh giá lại sau này mà không phải chạy lại.

Các checkpoint giữa chừng (`ckpt_round0109`, `ckpt_task02_FINAL`) đã gỡ vì lần
chạy đã xong, không còn cần resume. Chúng vẫn nằm trong lịch sử git nếu cần.
