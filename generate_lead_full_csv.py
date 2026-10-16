# import numpy as np
# import pandas as pd

# # ==============================================================================
# # 1. KHAI BÁO ĐƯỜNG DẪN DỮ LIỆU (4 LABEL CASA + 5 LABEL SAVING)
# # ==============================================================================
# # Đường dẫn 4 file score CASA
# casa_file_paths = {
#     "casa_100M": "./output_csv_score/score_CASA_S12_full_100M.csv",  # Hoặc file .xlsx
#     "casa_300M": "./output_csv_score/score_CASA_S12_full_300M.csv",
#     "casa_500M": "./output_csv_score/score_CASA_S12_full_500M.csv",
#     "casa_1B": "./output_csv_score/score_CASA_S12_full_1B.csv",
# }

# # Đường dẫn 5 file score SAVING (Tiết kiệm)
# saving_file_paths = {
#     "saving_new_open": "./output_csv_score/score_SAVING_S12_full_IS_NEW_OPEN.csv",
#     "saving_100M": "./output_csv_score/score_SAVING_S12_full_100M.csv",
#     "saving_300M": "./output_csv_score/score_SAVING_S12_full_300M.csv",
#     "saving_500M": "./output_csv_score/score_SAVING_S12_full_500M.csv",
#     "saving_1B": "./output_csv_score/score_SAVING_S12_full_1B.csv",
# }

# ID_COL = "CUSTOMER_CDE"
# SCORE_COL = "score"


# # ==============================================================================
# # 2. HÀM GÁN MỨC TIỀM NĂNG THEO QUY ĐỔI BỘ NHÃN MỚI
# # ==============================================================================
# def assign_tier(df, id_col=ID_COL, score_col=SCORE_COL):
#     """Sort score giảm dần và phân hạng theo Percentile:

#     - Top 0% - 5%: Cao
#     - Top >5% - 12%: Trung bình
#     - Top >12% - 25%: Thấp
#     - Ngoài Top 25%: Rất thấp
#     """
#     df_sorted = df[[id_col, score_col]].dropna().copy()
#     df_sorted = df_sorted.sort_values(by=score_col, ascending=False).reset_index(
#         drop=True
#     )

#     n = len(df_sorted)
#     if n == 0:
#         return df_sorted

#     # Tính Rank tỉ lệ
#     df_sorted["rank_pct"] = (df_sorted.index + 1) / n

#     # Gán Tier theo quy định mới
#     conditions = [
#         df_sorted["rank_pct"] <= 0.05,
#         (df_sorted["rank_pct"] > 0.05) & (df_sorted["rank_pct"] <= 0.12),
#         (df_sorted["rank_pct"] > 0.12) & (df_sorted["rank_pct"] <= 0.25),
#     ]
#     choices = ["Cao", "Trung bình", "Thấp"]

#     df_sorted["tier"] = np.select(conditions, choices, default="Rất thấp")
#     return df_sorted[[id_col, "tier"]]


# # Hàm phụ trợ để đọc và xử lý danh sách file
# def process_file_dict(file_dict, prefix_col_name=""):
#     processed_dfs = []
#     for label_name, path in file_dict.items():
#         if path.endswith(".xlsx") or path.endswith(".xls"):
#             df_raw = pd.read_excel(path)
#         else:
#             df_raw = pd.read_csv(path)

#         df_tier = assign_tier(df_raw)
#         col_title = f"{prefix_col_name}{label_name}".strip()
#         df_tier = df_tier.rename(columns={"tier": col_title})
#         processed_dfs.append(df_tier)
#     return processed_dfs


# # ==============================================================================
# # 3. ĐỌC VÀ TỔNG HỢP TOÀN BỘ 9 LABEL (4 CASA + 5 SAVING)
# # ==============================================================================
# # Xử lý 4 label CASA
# casa_dfs = process_file_dict(casa_file_paths, prefix_col_name="Tiềm năng ")
# casa_cols = [f"Tiềm năng {label}" for label in casa_file_paths.keys()]

# # Xử lý 5 label SAVING (CKH, 100M, 300M, 500M, 1B)
# saving_dfs = process_file_dict(
#     saving_file_paths, prefix_col_name="Tiềm năng SAVING_"
# )
# saving_cols = [f"Tiềm năng SAVING_{label}" for label in saving_file_paths.keys()]

# # Merge tất cả lại thành 1 DataFrame duy nhất
# all_dfs = casa_dfs + saving_dfs
# final_df = all_dfs[0]
# for df in all_dfs[1:]:
#     final_df = pd.merge(final_df, df, on=ID_COL, how="outer")

# # Điền 'Rất thấp' cho những KH không có dữ liệu score ở một số file
# all_label_cols = casa_cols + saving_cols
# final_df[all_label_cols] = final_df[all_label_cols].fillna("Rất thấp")


# # ==============================================================================
# # 4. TÍNH 2 CỘT TỔNG HỢP (CHỈ XÉT TRÊN 4 LABEL CASA)
# # ==============================================================================
# # Thứ tự ưu tiên Mức Tiềm Năng: Cao > Trung bình > Thấp > Rất thấp
# TIER_MAP = {"Cao": 4, "Trung bình": 3, "Thấp": 2, "Rất thấp": 1}
# MAP_TIER_BACK = {4: "Cao", 3: "Trung bình", 2: "Thấp", 1: "Rất thấp"}

# # Thứ tự ưu tiên Mốc CASA: 1B > 500M > 300M > 100M
# CASA_LABEL_ORDER = ["1B", "500M", "300M", "100M"]


# def get_max_casa_potential(row):
#     max_score = 1
#     best_label = "100M"

#     for label in CASA_LABEL_ORDER:
#         col_name = f"Tiềm năng casa_{label}"
#         tier_str = row[col_name]
#         numeric_val = TIER_MAP.get(tier_str, 1)

#         if numeric_val > max_score:
#             max_score = numeric_val
#             best_label = label

#     return pd.Series(
#         [MAP_TIER_BACK[max_score], best_label],
#         index=["Tiềm năng cao nhất là", "Tiềm năng cao nhất ở"],
#     )


# # Tính 2 cột tổng hợp dựa trên 4 cột CASA
# final_df[["Tiềm năng cao nhất là", "Tiềm năng cao nhất ở"]] = final_df.apply(
#     get_max_casa_potential, axis=1
# )


# # ==============================================================================
# # 5. SẮP XẾP LẠI THỨ TỰ CỘT VÀ XUẤT FILE FILE LEAD OUTPUT
# # ==============================================================================
# # Đổi tên 5 cột Saving cho đúng định dạng hiển thị đẹp
# saving_rename_dict = {
#     "Tiềm năng SAVING_saving_new_open": "Tiềm năng new open",
#     "Tiềm năng SAVING_saving_100M": "Tiềm năng 100M (Saving)",
#     "Tiềm năng SAVING_saving_300M": "Tiềm năng 300M (Saving)",
#     "Tiềm năng SAVING_saving_500M": "Tiềm năng 500M (Saving)",
#     "Tiềm năng SAVING_saving_1B": "Tiềm năng 1B (Saving)",
# }
# final_df = final_df.rename(columns=saving_rename_dict)
# renamed_saving_cols = list(saving_rename_dict.values())

# # Đặt thứ tự cột chuẩn: [CUSTOMER_CDE, 4 Cột CASA, 2 Cột Tổng Hợp, 5 Cột Saving]
# ordered_columns = (
#     [ID_COL]
#     + casa_cols
#     + ["Tiềm năng cao nhất là", "Tiềm năng cao nhất ở"]
#     + renamed_saving_cols
# )

# final_df = final_df[ordered_columns]

# # Lọc loại bỏ những KH có Tiềm năng CASA là "Rất thấp"
# lead_output = final_df[final_df["Tiềm năng cao nhất là"] != "Rất thấp"].copy()

# # Sort danh sách ưu tiên gửi Sale (Cao lên đầu)
# lead_output["_sort_rank"] = lead_output["Tiềm năng cao nhất là"].map(TIER_MAP)
# lead_output = lead_output.sort_values(
#     by=["_sort_rank", ID_COL], ascending=[False, True]
# ).drop(columns=["_sort_rank"])

# # Xuất ra file Excel
# output_filename = "Danh_Sach_Lead_Tong_Hop_CASA_Saving.csv"
# lead_output.to_csv(output_filename, index=False, encoding="utf-8-sig")

# print(
#     f"Đã hoàn thành! Xuất thành công {len(lead_output):,} KH ra file: {output_filename}"
# )


import numpy as np
import pandas as pd

# ==============================================================================
# 1. KHAI BÁO ĐƯỜNG DẪN DỮ LIỆU
# ==============================================================================
casa_file_paths = {
    "100M": "./output_csv_score/score_CASA_S12_full_100M.csv",
    "300M": "./output_csv_score/score_CASA_S12_full_300M.csv",
    "500M": "./output_csv_score/score_CASA_S12_full_500M.csv",
    "1B": "./output_csv_score/score_CASA_S12_full_1B.csv",
}

saving_file_paths = {
    "NEW_OPEN": "./output_csv_score/score_SAVING_S12_full_IS_NEW_OPEN.csv",
    "100M": "./output_csv_score/score_SAVING_S12_full_100M.csv",
    "300M": "./output_csv_score/score_SAVING_S12_full_300M.csv",
    "500M": "./output_csv_score/score_SAVING_S12_full_500M.csv",
    "1B": "./output_csv_score/score_SAVING_S12_full_1B.csv",
}

ID_COL = "CUSTOMER_CDE"
SCORE_COL = "score"


# ==============================================================================
# 2. HÀM GÁN MỨC TIỀM NĂNG THEO QUY ĐỔI PERCENTILE
# ==============================================================================
def assign_tier(df, id_col=ID_COL, score_col=SCORE_COL):
    """Sort score giảm dần và phân hạng theo Percentile:

    - Top 0% - 5%: Cao
    - Top >5% - 12%: Trung bình
    - Top >12% - 25%: Thấp
    - Ngoài Top 25%: Rất thấp
    """
    df_sorted = df[[id_col, score_col]].dropna().copy()
    df_sorted = df_sorted.sort_values(by=score_col, ascending=False).reset_index(
        drop=True
    )

    n = len(df_sorted)
    if n == 0:
        return df_sorted

    df_sorted["rank_pct"] = (df_sorted.index + 1) / n

    conditions = [
        df_sorted["rank_pct"] <= 0.05,
        (df_sorted["rank_pct"] > 0.05) & (df_sorted["rank_pct"] <= 0.12),
        (df_sorted["rank_pct"] > 0.12) & (df_sorted["rank_pct"] <= 0.25),
    ]
    choices = ["Cao", "Trung bình", "Thấp"]

    df_sorted["tier"] = np.select(conditions, choices, default="Rất thấp")
    return df_sorted[[id_col, "tier"]]


def process_file_dict(file_dict, prefix=""):
    processed_dfs = []
    for label_name, path in file_dict.items():
        if path.endswith(".xlsx") or path.endswith(".xls"):
            df_raw = pd.read_excel(path)
        else:
            df_raw = pd.read_csv(path)

        df_tier = assign_tier(df_raw)
        col_title = f"{prefix}{label_name}"
        df_tier = df_tier.rename(columns={"tier": col_title})
        processed_dfs.append(df_tier)
    return processed_dfs


# ==============================================================================
# 3. ĐỌC VÀ TỔNG HỢP TOÀN BỘ 9 LABEL (4 CASA + 5 SAVING)
# ==============================================================================
casa_dfs = process_file_dict(casa_file_paths, prefix="Tiềm năng CASA_")
saving_dfs = process_file_dict(saving_file_paths, prefix="Tiềm năng SAVING_")

casa_internal_cols = [
    f"Tiềm năng CASA_{k}" for k in casa_file_paths.keys()
]
saving_internal_cols = [
    f"Tiềm năng SAVING_{k}" for k in saving_file_paths.keys()
]

all_dfs = casa_dfs + saving_dfs
final_df = all_dfs[0]
for df in all_dfs[1:]:
    final_df = pd.merge(final_df, df, on=ID_COL, how="outer")

all_label_cols = casa_internal_cols + saving_internal_cols
final_df[all_label_cols] = final_df[all_label_cols].fillna("Rất thấp")


# ==============================================================================
# 4. TÍNH CÁC CỘT TỔNG HỢP
# ==============================================================================
TIER_MAP = {"Cao": 4, "Trung bình": 3, "Thấp": 2, "Rất thấp": 1}
MAP_TIER_BACK = {4: "Cao", 3: "Trung bình", 2: "Thấp", 1: "Rất thấp"}

# --- A. Tính 2 cột tổng hợp CASA ---
CASA_LABEL_ORDER = ["1B", "500M", "300M", "100M"]


def get_max_casa_potential(row):
    max_score = 1
    best_label = "100M"

    for label in CASA_LABEL_ORDER:
        col_name = f"Tiềm năng CASA_{label}"
        if col_name in row:
            tier_str = row[col_name]
            numeric_val = TIER_MAP.get(tier_str, 1)

            if numeric_val > max_score:
                max_score = numeric_val
                best_label = label

    return pd.Series(
        [MAP_TIER_BACK[max_score], best_label],
        index=["Tiềm năng cao nhất là", "Tiềm năng cao nhất ở"],
    )


final_df[["Tiềm năng cao nhất là", "Tiềm năng cao nhất ở"]] = final_df.apply(
    get_max_casa_potential, axis=1
)

# --- B. Tính cột "Tiềm năng Saving cao nhất ở" (100M/300M/500M/1B) ---
SAVING_LABEL_ORDER = ["1B", "500M", "300M", "100M"]


def get_max_saving_label(row):
    max_score = 1
    best_label = "100M"  # Default fallback

    for label in SAVING_LABEL_ORDER:
        col_name = f"Tiềm năng SAVING_{label}"
        if col_name in row:
            tier_str = row[col_name]
            numeric_val = TIER_MAP.get(tier_str, 1)

            if numeric_val > max_score:
                max_score = numeric_val
                best_label = label

    return best_label


final_df["Tiềm năng Saving cao nhất ở"] = final_df.apply(
    get_max_saving_label, axis=1
)


# ==============================================================================
# 5. ĐỔI TÊN CỘT ĐÚNG TEMPLATE VÀ XUẤT FILE
# ==============================================================================
column_rename_map = {
    "Tiềm năng CASA_100M": "Tiềm năng 100M",
    "Tiềm năng CASA_300M": "Tiềm năng 300M",
    "Tiềm năng CASA_500M": "Tiềm năng 500M",
    "Tiềm năng CASA_1B": "Tiềm năng 1B",
    "Tiềm năng SAVING_NEW_OPEN": "Tiềm năng new open",
    "Tiềm năng SAVING_100M": "Tiềm năng 100M (Saving)",
    "Tiềm năng SAVING_300M": "Tiềm năng 300M (Saving)",
    "Tiềm năng SAVING_500M": "Tiềm năng 500M (Saving)",
    "Tiềm năng SAVING_1B": "Tiềm năng 1B (Saving)",
}

final_df = final_df.rename(columns=column_rename_map)

# Thứ tự cột xuất ra file CSV
ordered_columns = [
    ID_COL,
    "Tiềm năng 100M",
    "Tiềm năng 300M",
    "Tiềm năng 500M",
    "Tiềm năng 1B",
    "Tiềm năng cao nhất là",
    "Tiềm năng cao nhất ở",
    "Tiềm năng new open",
    "Tiềm năng 100M (Saving)",
    "Tiềm năng 300M (Saving)",
    "Tiềm năng 500M (Saving)",
    "Tiềm năng 1B (Saving)",
    "Tiềm năng Saving cao nhất ở",  # Cột mới thêm
]

final_df = final_df[ordered_columns]

# Lọc loại bỏ những KH có Tiềm năng CASA là "Rất thấp"
lead_output = final_df[final_df["Tiềm năng cao nhất là"] != "Rất thấp"].copy()

# Sort danh sách ưu tiên gửi Sale
lead_output["_sort_rank"] = lead_output["Tiềm năng cao nhất là"].map(TIER_MAP)
lead_output = lead_output.sort_values(
    by=["_sort_rank", ID_COL], ascending=[False, True]
).drop(columns=["_sort_rank"])

# Xuất ra file CSV với chuẩn UTF-8-SIG để không bị lỗi font khi mở bằng Excel
output_filename = "Danh_Sach_Lead_Tong_Hop_CASA_Saving.csv"
lead_output.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(
    f"Đã hoàn thành! Xuất thành công {len(lead_output):,} KH ra file: {output_filename}"
)