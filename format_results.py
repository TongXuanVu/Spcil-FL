import pandas as pd
import re

def main():
    # Đọc file test_results.csv chứa toàn bộ kết quả từ Kaggle
    df = pd.read_csv("logs/der/cic_iot23/test_results.csv")
    
    clean_data = []
    
    for _, row in df.iterrows():
        ckpt = row['checkpoint']
        
        # Chỉ lấy các dòng checkpoint của từng round, bỏ qua các dòng _acc (checkpoint cuối task)
        match = re.search(r"ckpt_task(\d+)_round(\d+)\.pth", ckpt)
        if match:
            task_id = int(match.group(1))
            round_id = int(match.group(2))
            
            clean_data.append({
                "task_id": task_id,
                "round": round_id,
                "accuracy": round(float(row["acc"]), 2),
                "precision_micro": round(float(row["prec_mic"]), 2),
                "precision_macro": round(float(row["prec_mac"]), 2),
                "precision_weight": round(float(row["prec_wei"]), 2),
                "recall_micro": round(float(row["rec_mic"]), 2),
                "recall_macro": round(float(row["rec_mac"]), 2),
                "recall_weight": round(float(row["rec_wei"]), 2),
                "f1_micro": round(float(row["f1_mic"]), 2),
                "f1_macro": round(float(row["f1_mac"]), 2),
                "f1_weight": round(float(row["f1_wei"]), 2),
                "loss": round(float(row["loss"]), 4)
            })
            
    # Tạo DataFrame chuẩn gồm đúng 180 rounds
    df_clean = pd.DataFrame(clean_data)
    
    # Sắp xếp lại cho chắc chắn đúng thứ tự Task và Round
    df_clean = df_clean.sort_values(by=["task_id", "round"]).reset_index(drop=True)
    
    # Save file 180 rounds
    df_clean.to_csv("spcil_180_rounds.csv", index=False)
    print(f"Created spcil_180_rounds.csv with {len(df_clean)} rows.")
    
    # Extract Round 30 for summary
    df_final = df_clean[df_clean["round"] == 30].reset_index(drop=True)
    df_final.to_csv("spcil_final_rounds.csv", index=False)
    print(f"Created spcil_final_rounds.csv with {len(df_final)} rows.")

if __name__ == "__main__":
    main()
