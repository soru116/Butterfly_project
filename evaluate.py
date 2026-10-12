import pandas as pd

GT_CSV = "Perfect_Answer.csv"
PRED_CSV = "Test_Dataset.csv"


def main():
    df_gt = pd.read_csv(GT_CSV)
    df_pred = pd.read_csv(PRED_CSV)

    # 依 filename 對齊
    df = df_gt.merge(df_pred, on="filename",
                     suffixes=("_gt", "_pred"))

    TP = ((df.label_gt == 1) & (df.label_pred == 1)).sum()
    TN = ((df.label_gt == 0) & (df.label_pred == 0)).sum()
    FP = ((df.label_gt == 0) & (df.label_pred == 1)).sum()
    FN = ((df.label_gt == 1) & (df.label_pred == 0)).sum()

    accuracy = (TP + TN) / len(df)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0     # Sensitivity
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0     # Specificity

    print("===== Compare with Perfect_Answer.csv =====")
    print(f"TP = {TP}")
    print(f"TN = {TN}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print("----------------------------------")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"TPR (Sensitivity) = {TPR:.4f}")
    print(f"TNR (Specificity) = {TNR:.4f}")
    print("==========================================")


if __name__ == "__main__":
    main()
