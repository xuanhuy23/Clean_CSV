import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Dict, Union, Tuple, Optional, Any
import os
import logging
import sys
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import process_dataframe_in_batches

def missing_values_summary(df):
    """
    Tạo bảng tóm tắt về giá trị thiếu trong DataFrame
    
    Tham số:
    - df: DataFrame cần kiểm tra
    
    Trả về:
    - DataFrame: Tóm tắt về giá trị thiếu trong từng cột
    """
    # Tạo DataFrame tóm tắt
    missing_summary = pd.DataFrame({
        'Số lượng thiếu': df.isnull().sum(),
        'Tỷ lệ thiếu (%)': df.isnull().sum() * 100 / len(df)
    })
    
    # Sắp xếp theo tỷ lệ giá trị thiếu giảm dần
    missing_summary = missing_summary.sort_values('Tỷ lệ thiếu (%)', ascending=False)
    
    print("TÓM TẮT TÌNH TRẠNG GIÁ TRỊ THIẾU:")
    print(missing_summary)
    
    # Tính tổng số giá trị thiếu trong toàn bộ DataFrame
    total_missing = df.isnull().sum().sum()
    total_values = df.shape[0] * df.shape[1]
    total_missing_percent = total_missing * 100 / total_values
    
    print(f"\nTổng số giá trị thiếu: {total_missing}/{total_values} ({total_missing_percent:.2f}%)")
    
    return missing_summary

def visualize_missing_values(df, figsize=(12, 8)):
    """
    Trực quan hóa giá trị thiếu trong DataFrame
    
    Tham số:
    - df: DataFrame cần kiểm tra
    - figsize: Kích thước của biểu đồ
    """
    try:
        # Tạo heatmap hiển thị vị trí của giá trị thiếu
        plt.figure(figsize=figsize)
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Vị trí của giá trị thiếu trong dữ liệu')
        plt.tight_layout()
        plt.savefig('data/processed/missing_values_heatmap.png')
        plt.close()
        
        # Tạo biểu đồ cột hiển thị số lượng giá trị thiếu theo cột
        plt.figure(figsize=figsize)
        df.isnull().sum().sort_values(ascending=False).plot(kind='bar')
        plt.title('Số lượng giá trị thiếu theo cột')
        plt.ylabel('Số lượng')
        plt.tight_layout()
        plt.savefig('data/processed/missing_values_count.png')
        plt.close()
        
        print("Đã tạo biểu đồ trực quan hóa giá trị thiếu và lưu vào thư mục data/processed/")
    except Exception as e:
        print(f"Lỗi khi trực quan hóa giá trị thiếu: {str(e)}")

def delete_missing_values(df, axis=0, how='any', thresh=None, subset=None, inplace=False):
    """
    Xóa các hàng hoặc cột có giá trị thiếu
    
    Tham số:
    - df: DataFrame cần xử lý
    - axis: 0 để xóa hàng, 1 để xóa cột
    - how: 'any' (xóa nếu có bất kỳ giá trị NaN nào) hoặc 'all' (xóa chỉ khi tất cả là NaN)
    - thresh: Số lượng giá trị không phải NaN tối thiểu để giữ lại hàng/cột
    - subset: Danh sách các cột để xem xét (chỉ áp dụng khi axis=0)
    - inplace: True để thay đổi trực tiếp DataFrame gốc, False để trả về bản sao
    
    Trả về:
    - DataFrame đã xử lý
    """
    # Lưu kích thước ban đầu
    original_shape = df.shape
    
    # Thực hiện xóa - Bạn không thể sử dụng cả how và thresh
    if thresh is not None:
        # Ưu tiên sử dụng thresh nếu nó được cung cấp
        print(f"Sử dụng thresh={thresh} và bỏ qua how={how}")
        
        # Gọi dropna chỉ với thresh
        if inplace:
            df.dropna(axis=axis, thresh=thresh, subset=subset, inplace=True)
            result_df = df
        else:
            result_df = df.dropna(axis=axis, thresh=thresh, subset=subset, inplace=False)
    else:
        # Nếu thresh không được cung cấp, sử dụng how
        print(f"Sử dụng how={how}")
        
        # Gọi dropna chỉ với how
        if inplace:
            df.dropna(axis=axis, how=how, subset=subset, inplace=True)
            result_df = df
        else:
            result_df = df.dropna(axis=axis, how=how, subset=subset, inplace=False)
    
    # In thông báo về số lượng hàng/cột bị xóa
    if axis == 0:
        rows_removed = original_shape[0] - result_df.shape[0]
        print(f"Đã xóa {rows_removed} hàng có giá trị thiếu ({rows_removed/original_shape[0]*100:.2f}%)")
    else:
        cols_removed = original_shape[1] - result_df.shape[1]
        print(f"Đã xóa {cols_removed} cột có giá trị thiếu ({cols_removed/original_shape[1]*100:.2f}%)")
    
    return result_df

def fill_missing_values(df, strategy='simple', columns=None, inplace=False):
    """
    Điền giá trị thay thế cho giá trị thiếu
    
    Tham số:
    - df: DataFrame cần xử lý
    - strategy: Chiến lược điền giá trị thiếu
      - 'simple': Sử dụng giá trị đơn giản (0 cho số, '' cho chuỗi)
      - 'statistical': Sử dụng giá trị thống kê (mean/median cho số, mode cho chuỗi)
      - 'ffill': Điền tiến (forward fill)
      - 'bfill': Điền lùi (backward fill)
    - columns: Danh sách các cột cần xử lý (None để xử lý tất cả)
    - inplace: True để thay đổi trực tiếp DataFrame gốc, False để trả về bản sao
    
    Trả về:
    - DataFrame đã xử lý
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    # Xác định các cột cần xử lý
    if columns is None:
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        categorical_cols = result_df.select_dtypes(include=['object']).columns
    else:
        numeric_cols = [col for col in columns if col in result_df.select_dtypes(include=['number']).columns]
        categorical_cols = [col for col in columns if col in result_df.select_dtypes(include=['object']).columns]
    
    # Áp dụng chiến lược điền giá trị
    if strategy == 'simple':
        # Điền 0 cho các cột số
        for col in numeric_cols:
            result_df[col] = result_df[col].fillna(0)
            print(f"Đã điền giá trị 0 cho {result_df[col].isna().sum()} ô thiếu trong cột {col}")
        
        # Điền chuỗi rỗng cho các cột chuỗi
        for col in categorical_cols:
            result_df[col] = result_df[col].fillna('')
            print(f"Đã điền chuỗi rỗng cho {result_df[col].isna().sum()} ô thiếu trong cột {col}")
    
    elif strategy == 'statistical':
        # Điền mean hoặc median cho các cột số
        for col in numeric_cols:
            # Sử dụng median nếu dữ liệu có outlier (phân phối lệch)
            if abs(result_df[col].skew()) > 1:
                fill_value = result_df[col].median()
                print(f"Cột {col} có phân phối lệch, sử dụng median: {fill_value}")
            else:
                fill_value = result_df[col].mean()
                print(f"Cột {col} có phân phối cân đối, sử dụng mean: {fill_value}")
            
            missing_count = result_df[col].isna().sum()
            result_df[col] = result_df[col].fillna(fill_value)
            print(f"Đã điền giá trị {fill_value:.2f} cho {missing_count} ô thiếu trong cột {col}")
        
        # Điền mode cho các cột chuỗi
        for col in categorical_cols:
            if not result_df[col].empty and result_df[col].notna().any():
                mode_value = result_df[col].mode()[0]
                missing_count = result_df[col].isna().sum()
                result_df[col] = result_df[col].fillna(mode_value)
                print(f"Đã điền giá trị '{mode_value}' cho {missing_count} ô thiếu trong cột {col}")
    
    elif strategy == 'ffill':
        # Điền tiến
        for col in numeric_cols.union(categorical_cols):
            missing_count = result_df[col].isna().sum()
            result_df[col] = result_df[col].fillna(method='ffill')
            filled_count = missing_count - result_df[col].isna().sum()
            print(f"Đã điền tiến {filled_count} ô thiếu trong cột {col}")
    
    elif strategy == 'bfill':
        # Điền lùi
        for col in numeric_cols.union(categorical_cols):
            missing_count = result_df[col].isna().sum()
            result_df[col] = result_df[col].fillna(method='bfill')
            filled_count = missing_count - result_df[col].isna().sum()
            print(f"Đã điền lùi {filled_count} ô thiếu trong cột {col}")
    
    else:
        print(f"Chiến lược '{strategy}' không được hỗ trợ.")
    
    return result_df

def fill_missing_by_group(df, group_col, target_col, method='mean', inplace=False):
    """
    Điền giá trị thiếu dựa trên các nhóm
    
    Tham số:
    - df: DataFrame cần xử lý
    - group_col: Tên cột để nhóm dữ liệu
    - target_col: Tên cột cần điền giá trị thiếu
    - method: Phương pháp thống kê để áp dụng ('mean', 'median', 'mode')
    - inplace: True để thay đổi trực tiếp DataFrame gốc, False để trả về bản sao
    
    Trả về:
    - DataFrame đã xử lý
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    missing_count = result_df[target_col].isna().sum()
    
    # Kiểm tra xem cột nhóm và cột mục tiêu có tồn tại không
    if group_col not in result_df.columns:
        print(f"Cột nhóm '{group_col}' không tồn tại trong DataFrame.")
        return result_df
    
    if target_col not in result_df.columns:
        print(f"Cột mục tiêu '{target_col}' không tồn tại trong DataFrame.")
        return result_df
    
    # Áp dụng phương pháp thống kê tương ứng
    if method == 'mean':
        result_df[target_col] = result_df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.mean()))
    elif method == 'median':
        result_df[target_col] = result_df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.median()))
    elif method == 'mode':
        # Lưu ý: Mode có thể trả về nhiều giá trị, sử dụng giá trị đầu tiên
        result_df[target_col] = result_df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
    else:
        print(f"Phương pháp '{method}' không được hỗ trợ.")
        return result_df
    
    filled_count = missing_count - result_df[target_col].isna().sum()
    print(f"Đã điền {filled_count} ô thiếu trong cột '{target_col}' dựa trên nhóm '{group_col}' bằng phương pháp '{method}'")
    
    return result_df

def normalize_text(df: pd.DataFrame, columns: List[str] = None, inplace: bool = False, batch_size: int = 5000) -> pd.DataFrame:
    """
    Chuẩn hóa văn bản trong các cột được chỉ định, hỗ trợ xử lý theo lô.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    columns : List[str], optional
        Danh sách các cột cần chuẩn hóa. Nếu None, sẽ tự động chọn tất cả các cột kiểu chuỗi (object)
    inplace : bool, default False
        Nếu True, thay đổi trực tiếp DataFrame đầu vào. Nếu False, tạo bản sao
    batch_size : int, default 5000
        Kích thước mỗi lô khi xử lý
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột văn bản đã được chuẩn hóa
    """
    if not inplace:
        df = df.copy()
    
    # Chọn các cột để chuẩn hóa
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    else:
        # Chỉ giữ lại các cột tồn tại trong DataFrame
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        logging.warning("Không có cột nào để chuẩn hóa văn bản")
        return df
    
    def _normalize_text_batch(batch_df):
        """Hàm xử lý một lô dữ liệu"""
        result_df = batch_df.copy()
        
        for col in columns:
            if col in result_df.columns and result_df[col].dtype == 'object':
                # Loại bỏ khoảng trắng thừa
                result_df[col] = result_df[col].astype(str).str.strip()
        
                # Loại bỏ nhiều khoảng trắng liên tiếp
                result_df[col] = result_df[col].str.replace(r'\s+', ' ', regex=True)
        
                # Loại bỏ dấu câu thừa
                result_df[col] = result_df[col].str.replace(r'([.,!?])\1+', r'\1', regex=True)
        
                # Sửa lỗi viết hoa đầu câu
                result_df[col] = result_df[col].str.replace(r'^([a-z])', lambda m: m.group(1).upper(), regex=True)
        
                # Sửa lỗi thiếu khoảng trắng sau dấu câu
                result_df[col] = result_df[col].str.replace(r'([.,!?])([a-zA-Z])', r'\1 \2', regex=True)
        
        return result_df
    
    # Nếu DataFrame nhỏ hoặc batch_size <= 0, xử lý trực tiếp không chia lô
    if len(df) <= batch_size or batch_size <= 0:
        return _normalize_text_batch(df)
    
    # Sử dụng hàm xử lý theo lô
    return process_dataframe_in_batches(df, _normalize_text_batch, batch_size=batch_size)

def replace_values(df, column, replacements, regex=False, inplace=False):
    """
    Thay thế các giá trị cụ thể trong một cột
    
    Tham số:
    - df: DataFrame cần xử lý
    - column: Tên cột cần thực hiện thay thế
    - replacements: Dictionary ánh xạ giá trị cần thay thế thành giá trị mới
    - regex: True nếu sử dụng biểu thức chính quy
    - inplace: True để thay đổi trực tiếp DataFrame gốc, False để trả về bản sao
    
    Trả về:
    - DataFrame đã xử lý
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    if column not in result_df.columns:
        print(f"Cột '{column}' không tồn tại trong DataFrame.")
        return result_df
    
    # Lưu các giá trị duy nhất trước khi thay thế
    original_unique = result_df[column].unique() if not result_df[column].isna().all() else []
    
    # Thực hiện thay thế
    for old_value, new_value in replacements.items():
        if regex:
            result_df[column] = result_df[column].str.replace(old_value, new_value, regex=True)
        else:
            result_df[column] = result_df[column].replace(old_value, new_value)
    
    # Lưu các giá trị duy nhất sau khi thay thế
    new_unique = result_df[column].unique() if not result_df[column].isna().all() else []
    
    print(f"Thay thế giá trị trong cột '{column}':")
    print(f"- Số giá trị duy nhất trước khi thay thế: {len(original_unique)}")
    print(f"- Số giá trị duy nhất sau khi thay thế: {len(new_unique)}")
    
    if len(original_unique) > 0 and len(new_unique) > 0:
        print(f"- Thay đổi: {len(original_unique) - len(new_unique)} giá trị duy nhất")
    
    return result_df

def extract_pattern(df, column, pattern, new_columns, inplace=False):
    """
    Trích xuất thông tin từ một cột sử dụng biểu thức chính quy
    
    Tham số:
    - df: DataFrame cần xử lý
    - column: Tên cột cần trích xuất thông tin
    - pattern: Biểu thức chính quy để trích xuất
    - new_columns: Danh sách tên các cột mới sẽ lưu thông tin trích xuất
    - inplace: True để thay đổi trực tiếp DataFrame gốc, False để trả về bản sao
    
    Trả về:
    - DataFrame đã xử lý
    """
    if inplace:
        result_df = df
    else:
        result_df = df.copy()
    
    if column not in result_df.columns:
        print(f"Cột '{column}' không tồn tại trong DataFrame.")
        return result_df
    
    # Trích xuất thông tin bằng biểu thức chính quy
    extracted_df = result_df[column].str.extract(pattern)
    
    # Đặt tên cho các cột mới
    if len(new_columns) != extracted_df.shape[1]:
        print(f"Cảnh báo: Số lượng tên cột mới ({len(new_columns)}) không khớp với số lượng nhóm trích xuất ({extracted_df.shape[1]})")
        new_columns = new_columns[:extracted_df.shape[1]]
    
    extracted_df.columns = new_columns[:extracted_df.shape[1]]
    
    # Thêm các cột mới vào DataFrame kết quả
    for col in extracted_df.columns:
        result_df[col] = extracted_df[col]
    
    print(f"Đã trích xuất thông tin từ cột '{column}' thành {extracted_df.shape[1]} cột mới:")
    for col in extracted_df.columns:
        non_null_count = result_df[col].notna().sum()
        print(f"- Cột '{col}': {non_null_count} giá trị không thiếu")
    
    return result_df

# ============= TASK 2.3: CHUẨN HÓA NGÀY THÁNG =============

def standardize_dates(df: pd.DataFrame, column: str, format: str = None, errors: str = 'raise') -> pd.DataFrame:
    """
    Chuẩn hóa cột ngày tháng trong DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    column : str
        Tên cột cần chuẩn hóa
    format : str, optional
        Định dạng ngày tháng, ví dụ: '%Y-%m-%d'
        Nếu là None, hàm sẽ tự động phát hiện định dạng phổ biến nhất
    errors : str, default 'raise'
        Xử lý khi gặp lỗi chuyển đổi: 'raise', 'coerce', 'ignore'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với cột ngày tháng đã được chuẩn hóa
    """
    if column not in df.columns:
        print(f"Cảnh báo: Cột '{column}' không tồn tại trong DataFrame")
        return df
        
    # Kiểm tra tên cột xem có khả năng là ngày tháng không
    name_keywords = ['name', 'station', 'address', 'street', 'city', 'state', 'zip', 'description', 'comment', 'text']
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'timestamp']
    
    # Kiểm tra tên cột - nếu chứa từ khóa không phải ngày tháng và không chứa từ khóa ngày tháng
    if any(keyword in column.lower() for keyword in name_keywords) and not any(keyword in column.lower() for keyword in date_keywords):
        print(f"Cảnh báo: Cột '{column}' có tên gợi ý không phải dạng ngày tháng. Bỏ qua việc chuyển đổi.")
        return df
    
    # Tạo bản sao để không thay đổi DataFrame gốc
    result_df = df.copy()
    
    # Kiểm tra nếu cột đã là kiểu datetime
    if pd.api.types.is_datetime64_any_dtype(result_df[column]):
        print(f"Cột '{column}' đã là kiểu ngày tháng: {result_df[column].dtype}")
        return result_df
    
    # Kiểm tra dữ liệu mẫu
    sample_values = df[column].dropna().astype(str).head(50).tolist()
    if sample_values:
        # Kiểm tra xem mẫu có phải là văn bản dài hoặc phức tạp không
        if any(len(str(x)) > 30 for x in sample_values) or any(str(x).count(' ') > 3 for x in sample_values):
            print(f"Cảnh báo: Dữ liệu trong cột '{column}' có vẻ là văn bản dài/phức tạp, không phải ngày tháng. Bỏ qua việc chuyển đổi.")
            return df
        
        # Đếm số giá trị mẫu có thể chuyển đổi thành datetime
        convertible_count = 0
        for value in sample_values:
            try:
                pd.to_datetime(value, errors='raise')
                convertible_count += 1
            except:
                pass
        
        # Nếu ít hơn 30% mẫu có thể chuyển đổi, đây không phải là cột ngày tháng
        if convertible_count < 0.3 * len(sample_values):
            print(f"Cảnh báo: Chỉ {convertible_count}/{len(sample_values)} giá trị mẫu có thể chuyển đổi thành ngày tháng. Bỏ qua việc chuyển đổi.")
            return df
    
    # Thử tự động phát hiện định dạng nếu không được chỉ định
    if format is None:
        try:
            # Import function từ data_loader nếu có thể
            from data_loader import detect_common_date_format
            detected_format = detect_common_date_format(df, column)
            if detected_format:
                format = detected_format
                print(f"Đã phát hiện định dạng ngày tháng cho cột '{column}': {format}")
        except (ImportError, ModuleNotFoundError):
            # Thử lấy định dạng từ module hiện tại nếu có
            try:
                # Lấy mẫu các giá trị không null
                if sample_values:
                    print(f"Các giá trị mẫu từ cột '{column}': {sample_values[:5]}")
                    
                    # Thử một số định dạng phổ biến
                    common_formats = [
                        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
                        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
                        '%b %d, %Y', '%d %b %Y', '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %H:%M:%S', 
                        '%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S',
                        '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M',
                        '%d-%m-%Y %H:%M'
                    ]
                    
                    # Đếm số lần thành công cho mỗi định dạng
                    format_counts = {}
                    
                    for test_format in common_formats:
                        success_count = 0
                        for value in sample_values:
                            try:
                                if pd.to_datetime(value, format=test_format):
                                    success_count += 1
                            except (ValueError, TypeError):
                                continue
                        
                        if success_count > 0:
                            format_counts[test_format] = success_count
                    
                    # Chọn định dạng phù hợp với nhiều giá trị nhất
                    if format_counts:
                        format = max(format_counts.items(), key=lambda x: x[1])[0]
                        success_rate = format_counts[format] / len(sample_values) * 100
                        print(f"Đã phát hiện định dạng '{format}' phù hợp với {format_counts[format]}/{len(sample_values)} ({success_rate:.2f}%) giá trị mẫu")
                        
                        # Nếu tỷ lệ thành công thấp, không nên tiếp tục
                        if success_rate < 50:
                            print(f"Cảnh báo: Tỷ lệ chuyển đổi thành công quá thấp ({success_rate:.2f}%). Bỏ qua việc chuyển đổi.")
                            return df
            except Exception as e:
                print(f"Không thể tự động phát hiện định dạng: {e}")
                return df
    
    try:
        # Đếm số lượng giá trị không null trước khi chuyển đổi
        non_null_count = result_df[column].notna().sum()
        
        # Thực hiện chuyển đổi
        if format:
            # Thử chuyển đổi với định dạng đã phát hiện
            result_df[column] = pd.to_datetime(result_df[column], format=format, errors=errors)
            print(f"Đã áp dụng định dạng '{format}' cho cột '{column}'")
        else:
            # Nếu không phát hiện được định dạng cụ thể, sử dụng chuyển đổi tự động
            print(f"Không phát hiện được định dạng cụ thể cho cột '{column}', áp dụng chuyển đổi tự động")
            result_df[column] = pd.to_datetime(result_df[column], errors=errors)
        
        # Đếm số lượng giá trị không null sau khi chuyển đổi
        converted_count = result_df[column].notna().sum()
        
        # Kiểm tra có bao nhiêu giá trị bị chuyển thành NaT
        if converted_count < non_null_count:
            nat_count = non_null_count - converted_count
            nat_percent = (nat_count / non_null_count) * 100
            print(f"Cảnh báo: {nat_count}/{non_null_count} ({nat_percent:.2f}%) giá trị không thể chuyển đổi và trở thành NaT")
            
            # Nếu quá nhiều giá trị không thể chuyển đổi, có thể đây không phải cột ngày tháng
            if nat_percent > 70:
                print(f"Cảnh báo: Tỷ lệ chuyển đổi thất bại quá cao ({nat_percent:.2f}%). Có thể cột '{column}' không phải kiểu ngày tháng.")
                if errors != 'raise':
                    print("Trả về DataFrame gốc không thay đổi.")
                    return df
            
            # Hiển thị một số giá trị không thể chuyển đổi
            if nat_count > 0 and 'format' in locals() and format:
                # Xác định các giá trị không thể chuyển đổi
                temp_df = df.copy()
                try:
                    temp_df['_temp_col'] = pd.to_datetime(temp_df[column], format=format, errors='coerce')
                    invalid_mask = temp_df['_temp_col'].isna() & temp_df[column].notna()
                    invalid_examples = temp_df.loc[invalid_mask, column].astype(str).head(5).tolist()
                    if invalid_examples:
                        print(f"Ví dụ về giá trị không thể chuyển đổi: {invalid_examples}")
                except:
                    pass
        
        # In thông tin về kiểu dữ liệu mới
        print(f"Đã chuyển đổi cột '{column}' sang định dạng {result_df[column].dtype}")
        return result_df
        
    except Exception as e:
        if errors == 'raise':
            print(f"Lỗi khi chuẩn hóa cột ngày tháng '{column}': {str(e)}")
            raise e
        elif errors == 'ignore':
            print(f"Cảnh báo: Không thể chuẩn hóa cột ngày tháng '{column}': {str(e)}")
        return result_df

def extract_datetime_components(df: pd.DataFrame, date_column: str, 
                               components: List[str] = ['year', 'month', 'day'],
                               prefix: str = None) -> pd.DataFrame:
    """
    Trích xuất các thành phần ngày tháng từ cột datetime và thêm vào DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    date_column : str
        Tên cột chứa dữ liệu ngày tháng (phải là kiểu datetime)
    components : List[str], default ['year', 'month', 'day']
        Các thành phần cần trích xuất ('year', 'month', 'day', 'hour', 'minute', 
        'second', 'dayofweek', 'dayofyear', 'quarter', 'week')
    prefix : str, optional
        Tiền tố cho tên cột mới. Nếu None, sẽ sử dụng tên cột gốc
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột mới chứa các thành phần ngày tháng
    """
    if date_column not in df.columns:
        print(f"Không tìm thấy cột '{date_column}' trong DataFrame")
        return df
    
    result_df = df.copy()
    
    # Kiểm tra xem cột đã ở định dạng datetime chưa
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        print(f"Cảnh báo: Cột '{date_column}' không phải là kiểu datetime. Đang thử chuyển đổi...")
        result_df = standardize_dates(result_df, date_column, errors='coerce')
        
        # Kiểm tra lại sau khi chuyển đổi
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
            print(f"Không thể chuyển đổi cột '{date_column}' sang định dạng datetime. Không thể trích xuất thành phần.")
            return df
    
    # Xác định tiền tố cho tên cột mới
    prefix = prefix or f"{date_column}_"
    
    # Trích xuất các thành phần được chỉ định
    for component in components:
        try:
            if component == 'year':
                result_df[f"{prefix}year"] = result_df[date_column].dt.year
                print(f"Đã tạo cột {prefix}year")
            elif component == 'month':
                result_df[f"{prefix}month"] = result_df[date_column].dt.month
                print(f"Đã tạo cột {prefix}month")
            elif component == 'day':
                result_df[f"{prefix}day"] = result_df[date_column].dt.day
                print(f"Đã tạo cột {prefix}day")
            elif component == 'hour':
                result_df[f"{prefix}hour"] = result_df[date_column].dt.hour
                print(f"Đã tạo cột {prefix}hour")
            elif component == 'minute':
                result_df[f"{prefix}minute"] = result_df[date_column].dt.minute
                print(f"Đã tạo cột {prefix}minute")
            elif component == 'second':
                result_df[f"{prefix}second"] = result_df[date_column].dt.second
                print(f"Đã tạo cột {prefix}second")
            elif component == 'dayofweek':
                result_df[f"{prefix}dayofweek"] = result_df[date_column].dt.dayofweek
                print(f"Đã tạo cột {prefix}dayofweek (0=Monday, 6=Sunday)")
            elif component == 'dayofyear':
                result_df[f"{prefix}dayofyear"] = result_df[date_column].dt.dayofyear
                print(f"Đã tạo cột {prefix}dayofyear")
            elif component == 'quarter':
                result_df[f"{prefix}quarter"] = result_df[date_column].dt.quarter
                print(f"Đã tạo cột {prefix}quarter")
            elif component == 'week':
                result_df[f"{prefix}week"] = result_df[date_column].dt.isocalendar().week
                print(f"Đã tạo cột {prefix}week")
            else:
                print(f"Thành phần không được hỗ trợ: '{component}'")
        except Exception as e:
            print(f"Lỗi khi trích xuất thành phần '{component}': {str(e)}")
    
    return result_df

def format_datetime(df: pd.DataFrame, column: str, format_str: str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Định dạng lại cột ngày tháng thành chuỗi theo định dạng được chỉ định
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    column : str
        Tên cột chứa dữ liệu ngày tháng
    format_str : str, default '%Y-%m-%d'
        Định dạng đầu ra sử dụng cú pháp strftime
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với cột đã được định dạng lại
    """
    if column not in df.columns:
        print(f"Không tìm thấy cột '{column}' trong DataFrame")
        return df
    
    result_df = df.copy()
    
    # Kiểm tra xem cột đã ở định dạng datetime chưa
    if not pd.api.types.is_datetime64_any_dtype(result_df[column]):
        print(f"Cảnh báo: Cột '{column}' không phải là kiểu datetime. Đang thử chuyển đổi...")
        result_df = standardize_dates(result_df, column, errors='coerce')
        
        # Kiểm tra lại sau khi chuyển đổi
        if not pd.api.types.is_datetime64_any_dtype(result_df[column]):
            print(f"Không thể chuyển đổi cột '{column}' sang định dạng datetime. Không thể định dạng lại.")
            return df
    
    # Định dạng lại thành chuỗi
    try:
        result_df[column] = result_df[column].dt.strftime(format_str)
        print(f"Đã định dạng lại cột '{column}' theo định dạng '{format_str}'")
    except Exception as e:
        print(f"Lỗi khi định dạng lại cột '{column}': {str(e)}")
        return df
    
    return result_df

# ============= TASK 2.4: CHUYỂN ĐỔI KIỂU DỮ LIỆU =============

def convert_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Chuyển đổi kiểu dữ liệu của nhiều cột dựa trên ánh xạ kiểu được cung cấp
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    type_mapping : Dict[str, str]
        Từ điển ánh xạ từ tên cột sang kiểu dữ liệu mong muốn
        Ví dụ: {'age': 'int64', 'salary': 'float64', 'category': 'category'}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với các kiểu dữ liệu đã được chuyển đổi
    """
    result_df = df.copy()
    
    for column, dtype in type_mapping.items():
        if column not in result_df.columns:
            print(f"Cảnh báo: Không tìm thấy cột '{column}' trong DataFrame")
            continue
        
        try:
            # Lưu kiểu dữ liệu hiện tại để so sánh
            original_dtype = result_df[column].dtype
            
            # Chuyển đổi kiểu dữ liệu
            result_df[column] = result_df[column].astype(dtype)
            
            print(f"Đã chuyển đổi cột '{column}' từ {original_dtype} sang {dtype}")
        except Exception as e:
            print(f"Lỗi khi chuyển đổi cột '{column}' sang kiểu {dtype}: {str(e)}")
    
    return result_df

def convert_to_numeric(df: pd.DataFrame, columns: List[str], errors: str = 'raise') -> pd.DataFrame:
    """
    Chuyển đổi các cột được chỉ định sang kiểu số
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    columns : List[str]
        Danh sách tên cột cần chuyển đổi
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - 'raise': báo lỗi nếu không thể chuyển đổi
        - 'coerce': chuyển đổi thành NaN nếu không thể chuyển đổi
        - 'ignore': giữ nguyên giá trị nếu không thể chuyển đổi
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột đã được chuyển đổi sang kiểu số
    """
    result_df = df.copy()
    
    for column in columns:
        if column not in result_df.columns:
            print(f"Cảnh báo: Không tìm thấy cột '{column}' trong DataFrame")
            continue
        
        # Kiểm tra xem cột đã là kiểu số chưa
        if pd.api.types.is_numeric_dtype(result_df[column]):
            print(f"Cột '{column}' đã là kiểu số: {result_df[column].dtype}")
            continue
        
        try:
            # Xử lý định dạng tiền tệ, loại bỏ ký tự '$', ',', etc.
            if result_df[column].dtype == 'object':
                # Hiển thị một số giá trị mẫu
                n_samples = min(5, len(result_df[column].dropna()))
                sample_values = result_df[column].dropna().iloc[:n_samples].values
                print(f"Các giá trị mẫu từ cột '{column}': {sample_values}")
                
                # Loại bỏ các ký tự đặc biệt và dấu phân cách hàng nghìn
                result_df[column] = result_df[column].astype(str).str.replace('$', '', regex=False)
                result_df[column] = result_df[column].str.replace(',', '', regex=False)
                result_df[column] = result_df[column].str.replace(r'[^\d.-]', '', regex=True)
                
                # Chuyển đổi sang kiểu số
                result_df[column] = pd.to_numeric(result_df[column], errors=errors)
            else:
                # Nếu không phải chuỗi, thử chuyển đổi trực tiếp
                result_df[column] = pd.to_numeric(result_df[column], errors=errors)
            
            print(f"Đã chuyển đổi cột '{column}' thành kiểu số: {result_df[column].dtype}")
            
            # Báo cáo số lượng NaN sau khi chuyển đổi nếu errors='coerce'
            if errors == 'coerce':
                nan_count = result_df[column].isna().sum()
                total_count = len(result_df[column])
                if nan_count > 0:
                    print(f"Cảnh báo: {nan_count}/{total_count} ({nan_count*100/total_count:.2f}%) giá trị không thể chuyển đổi và trở thành NaN")
        
        except Exception as e:
            print(f"Lỗi khi chuyển đổi cột '{column}' sang kiểu số: {str(e)}")
            if errors == 'raise':
                raise e
    
    return result_df

def convert_to_category(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Chuyển đổi các cột được chỉ định sang kiểu category
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    columns : List[str]
        Danh sách tên cột cần chuyển đổi
    
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột đã được chuyển đổi sang kiểu category
    """
    result_df = df.copy()
    
    for column in columns:
        if column not in result_df.columns:
            print(f"Cảnh báo: Không tìm thấy cột '{column}' trong DataFrame")
            continue
        
        # Kiểm tra xem cột đã là kiểu category chưa
        if pd.api.types.is_categorical_dtype(result_df[column]):
            print(f"Cột '{column}' đã là kiểu category")
            continue
        
        try:
            # Đếm số lượng giá trị duy nhất
            n_unique = result_df[column].nunique()
            n_total = len(result_df[column])
            pct_unique = (n_unique / n_total) * 100
            
            # Hiển thị thông tin về số lượng giá trị duy nhất
            print(f"Cột '{column}' có {n_unique} giá trị duy nhất / {n_total} giá trị ({pct_unique:.2f}%)")
            
            # Chỉ chuyển sang category nếu số lượng giá trị duy nhất đủ nhỏ
            if pct_unique <= 50:  # Ngưỡng: có ít hơn 50% giá trị duy nhất
                result_df[column] = result_df[column].astype('category')
                print(f"Đã chuyển đổi cột '{column}' sang kiểu category")
                
                # Hiển thị các giá trị duy nhất nếu số lượng nhỏ
                if n_unique <= 10:
                    unique_values = result_df[column].cat.categories.tolist()
                    print(f"Các giá trị phân loại: {unique_values}")
            else:
                print(f"Không chuyển đổi cột '{column}' vì số lượng giá trị duy nhất quá lớn")
        
        except Exception as e:
            print(f"Lỗi khi chuyển đổi cột '{column}' sang kiểu category: {str(e)}")
    
    return result_df

# ============= TASK 2.5: XỬ LÝ DỮ LIỆU TRÙNG LẶP =============

def detect_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Dict[str, Any]:
    """
    Phát hiện dữ liệu trùng lặp trong DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần kiểm tra
    subset : List[str], optional
        Danh sách các cột cần kiểm tra trùng lặp. Nếu None, sẽ kiểm tra toàn bộ cột.
    
    Returns:
    --------
    Dict[str, Any]
        Từ điển chứa thông tin về dữ liệu trùng lặp:
        - 'duplicate_count': số dòng trùng lặp
        - 'duplicate_percent': tỷ lệ phần trăm dòng trùng lặp
        - 'first_duplicates': các dòng trùng lặp đầu tiên (tối đa 5 dòng)
    """
    # Tìm dữ liệu trùng lặp
    duplicated = df.duplicated(subset=subset, keep='first')
    duplicate_count = duplicated.sum()
    duplicate_percent = (duplicate_count / len(df)) * 100
    
    # In thông tin về dữ liệu trùng lặp
    if subset:
        print(f"Kiểm tra trùng lặp trong {len(subset)} cột: {subset}")
    else:
        print("Kiểm tra trùng lặp trong toàn bộ DataFrame")
    
    print(f"Số dòng trùng lặp: {duplicate_count}/{len(df)} ({duplicate_percent:.2f}%)")
    
    # Lấy các dòng trùng lặp
    duplicated_rows = df[duplicated]
    
    # Nếu có dữ liệu trùng lặp, hiển thị một số ví dụ
    if duplicate_count > 0:
        # Lấy tối đa 5 dòng trùng lặp đầu tiên để hiển thị
        first_duplicates = duplicated_rows.head(5)
        print("\nCác dòng trùng lặp đầu tiên:")
        print(first_duplicates)
        
        # Nếu có các cột chỉ định, hiển thị giá trị của các cột đó
        if subset:
            print("\nGiá trị của các cột được kiểm tra:")
            for i, row in first_duplicates.iterrows():
                print(f"Dòng {i}: {row[subset].to_dict()}")
    
    # Trả về thông tin về dữ liệu trùng lặp
    return {
        'duplicate_count': duplicate_count,
        'duplicate_percent': duplicate_percent,
        'first_duplicates': duplicated_rows.head(5) if duplicate_count > 0 else None
    }

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Loại bỏ dữ liệu trùng lặp từ DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    subset : List[str], optional
        Danh sách các cột cần kiểm tra trùng lặp. Nếu None, sẽ kiểm tra toàn bộ cột.
    keep : {'first', 'last', False}, default 'first'
        - 'first': giữ lại dòng đầu tiên của mỗi nhóm trùng lặp
        - 'last': giữ lại dòng cuối cùng của mỗi nhóm trùng lặp
        - False: loại bỏ tất cả các dòng trùng lặp
    
    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi đã loại bỏ dữ liệu trùng lặp
    """
    # Đếm số dòng ban đầu
    original_count = len(df)
    
    # Loại bỏ dữ liệu trùng lặp
    result_df = df.drop_duplicates(subset=subset, keep=keep)
    
    # Đếm số dòng đã loại bỏ
    removed_count = original_count - len(result_df)
    removed_percent = (removed_count / original_count) * 100
    
    # In thông tin về việc loại bỏ dữ liệu trùng lặp
    if subset:
        column_desc = f"các cột: {subset}"
    else:
        column_desc = "toàn bộ cột"
        
    keep_desc = {
        'first': "dòng đầu tiên",
        'last': "dòng cuối cùng",
        False: "không giữ lại dòng nào"
    }.get(keep, keep)
    
    print(f"Đã loại bỏ {removed_count}/{original_count} dòng trùng lặp ({removed_percent:.2f}%), dựa trên {column_desc}, giữ lại {keep_desc}")
    print(f"Số dòng sau khi loại bỏ trùng lặp: {len(result_df)}")
    
    return result_df

# ============= TASK 2.6: XỬ LÝ GIÁ TRỊ NGOẠI LAI =============

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Phát hiện giá trị ngoại lai trong một cột
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu cần xử lý
    column : str
        Tên cột cần kiểm tra giá trị ngoại lai
    method : str, default 'zscore'
        Phương pháp phát hiện giá trị ngoại lai:
        - 'zscore': sử dụng z-score (khoảng cách số độ lệch chuẩn từ giá trị trung bình)
        - 'iqr': sử dụng IQR (Interquartile Range)
    threshold : float, default 3.0
        Ngưỡng để xác định giá trị ngoại lai:
        - Nếu method='zscore': số độ lệch chuẩn từ giá trị trung bình (thường là 3)
        - Nếu method='iqr': số lần IQR từ Q1 và Q3 (thường là 1.5)
    
    Returns:
    --------
    pd.Series
        Series boolean chỉ ra vị trí của các giá trị ngoại lai (True cho giá trị ngoại lai)
    """
    if column not in df.columns:
        print(f"Không tìm thấy cột '{column}' trong DataFrame")
        return pd.Series()
    
    # Kiểm tra xem cột có phải là kiểu số không
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Cảnh báo: Cột '{column}' không phải là kiểu số. Phát hiện giá trị ngoại lai chỉ áp dụng cho dữ liệu số.")
        return pd.Series()
    
    # Loại bỏ giá trị NaN
    series = df[column].dropna()
    
    # Phát hiện giá trị ngoại lai
    outliers = pd.Series(False, index=df.index)
    
    if method == 'zscore':
        # Sử dụng z-score
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            print(f"Cảnh báo: Độ lệch chuẩn của cột '{column}' bằng 0, không thể tính z-score")
            return outliers
        
        zscore = (df[column] - mean) / std
        outliers = zscore.abs() > threshold
        
        # In thông tin về phân phối
        print(f"Phương pháp Z-Score cho cột '{column}':")
        print(f"- Giá trị trung bình: {mean:.4f}")
        print(f"- Độ lệch chuẩn: {std:.4f}")
        print(f"- Ngưỡng Z-Score: +/- {threshold:.1f}")
        
    elif method == 'iqr':
        # Sử dụng IQR (Interquartile Range)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            print(f"Cảnh báo: IQR của cột '{column}' bằng 0, không thể phát hiện giá trị ngoại lai")
            return outliers
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        # In thông tin về phân phối
        print(f"Phương pháp IQR cho cột '{column}':")
        print(f"- Q1 (25%): {q1:.4f}")
        print(f"- Q3 (75%): {q3:.4f}")
        print(f"- IQR: {iqr:.4f}")
        print(f"- Ngưỡng: Dưới {lower_bound:.4f} hoặc trên {upper_bound:.4f}")
        
    else:
        print(f"Phương pháp '{method}' không được hỗ trợ. Sử dụng 'zscore' hoặc 'iqr'.")
        return outliers
    
    # Thống kê về giá trị ngoại lai
    outlier_count = outliers.sum()
    outlier_percent = (outlier_count / len(df)) * 100
    print(f"Phát hiện {outlier_count}/{len(df)} giá trị ngoại lai ({outlier_percent:.2f}%)")
    
    # Hiển thị một số giá trị ngoại lai
    if outlier_count > 0:
        outlier_values = df.loc[outliers, column]
        n_examples = min(5, outlier_count)
        print(f"\n{n_examples} ví dụ về giá trị ngoại lai:")
        print(outlier_values.head(n_examples))
        
        # Tính các thống kê cho giá trị ngoại lai
        print("\nThống kê cho giá trị ngoại lai:")
        print(f"- Min: {outlier_values.min():.4f}")
        print(f"- Max: {outlier_values.max():.4f}")
        print(f"- Mean: {outlier_values.mean():.4f}")
    
    return outliers

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'clip', 
                    threshold: float = 3.0, approach: str = 'zscore', batch_size: int = 5000) -> pd.DataFrame:
    """
    Xử lý các giá trị ngoại lai (outliers) trong một cột số, hỗ trợ xử lý theo lô.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    column : str
        Tên cột cần xử lý outliers
    method : str, default 'clip'
        Phương pháp xử lý outliers: 'remove' (xóa), 'clip' (cắt về giới hạn), 'iqr' (lọc IQR), 'zscore' (lọc Z-score)
    threshold : float, default 3.0
        Ngưỡng xác định outliers (z-score hoặc số lần IQR)
    approach : str, default 'zscore'
        Phương pháp xác định outliers: 'zscore' hoặc 'iqr'
    batch_size : int, default 5000
        Kích thước mỗi lô khi xử lý
    
    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi xử lý outliers
    """
    if column not in df.columns:
        logging.warning(f"Cột '{column}' không tồn tại trong DataFrame")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        logging.warning(f"Cột '{column}' không phải kiểu số, không thể xử lý outliers")
        return df
    
    # Hàm xác định giới hạn outliers
    def _get_outlier_bounds(data_series):
        if approach == 'zscore':
            mean = data_series.mean()
            std = data_series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        elif approach == 'iqr':
            q1 = data_series.quantile(0.25)
            q3 = data_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
        else:
            # Mặc định nếu phương pháp không hợp lệ
            lower_bound = data_series.min()
            upper_bound = data_series.max()
            
        return lower_bound, upper_bound
    
    # Tính toán giới hạn dựa trên toàn bộ dữ liệu
    non_null_data = df[column].dropna()
    if len(non_null_data) == 0:
        logging.warning(f"Cột '{column}' chỉ chứa giá trị null, không thể xử lý outliers")
        return df
        
    lower_bound, upper_bound = _get_outlier_bounds(non_null_data)
    
    # Hàm xử lý outliers cho từng lô
    def _handle_outliers_batch(batch_df):
        result_df = batch_df.copy()
        
        if method == 'remove':
            # Xóa các dòng có giá trị ngoại lai
            mask = (result_df[column] >= lower_bound) & (result_df[column] <= upper_bound)
            result_df = result_df[mask | result_df[column].isna()]
        elif method == 'clip':
            # Cắt giới hạn các giá trị ngoại lai
            result_df[column] = result_df[column].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'cap':
            # Tương tự clip, nhưng đặt tên khác
            result_df[column] = result_df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return result_df
    
    # Nếu DataFrame nhỏ hoặc batch_size <= 0, xử lý trực tiếp
    if len(df) <= batch_size or batch_size <= 0:
        return _handle_outliers_batch(df)
        
    # Sử dụng hàm xử lý theo lô
    return process_dataframe_in_batches(df, _handle_outliers_batch, batch_size=batch_size)
    
def validate_output(input_file: str, output_file: str, encoding: str = 'utf-8') -> bool:
    """
    Kiểm tra tính hợp lệ của dữ liệu đầu ra bằng cách so sánh với dữ liệu đầu vào.
    
    Parameters:
    -----------
    input_file : str
        Đường dẫn đến file CSV đầu vào gốc
    output_file : str
        Đường dẫn đến file CSV đầu ra đã làm sạch
    encoding : str, default 'utf-8'
        Mã hóa ký tự của file
        
    Returns:
    --------
    bool
        True nếu dữ liệu đầu ra hợp lệ, False nếu có lỗi
    """
    try:
        # Kiểm tra file đầu ra có tồn tại không
        if not os.path.exists(output_file):
            logging.error(f"File đầu ra không tồn tại: {output_file}")
            print(f"Lỗi: File đầu ra không tồn tại. Vui lòng kiểm tra lại quá trình xử lý.")
            return False
            
        # Kiểm tra kích thước file đầu ra
        if os.path.getsize(output_file) == 0:
            logging.error(f"File đầu ra rỗng: {output_file}")
            print(f"Lỗi: File đầu ra rỗng. Có thể đã xảy ra lỗi trong quá trình lưu.")
            return False
            
        print(f"Đang kiểm tra tính hợp lệ của file đầu ra...")
        logging.info(f"Đang kiểm tra tính hợp lệ của file đầu ra: {output_file}")
        
        # Đọc dữ liệu đầu vào và đầu ra
        try:
            df_input = pd.read_csv(input_file, encoding=encoding, low_memory=False)
            logging.info(f"Đã đọc file đầu vào: {input_file}, kích thước: {df_input.shape}")
        except Exception as e:
            logging.error(f"Không thể đọc file đầu vào: {str(e)}")
            print(f"Lỗi: Không thể đọc file đầu vào. Chi tiết: {str(e)}")
            # Vẫn tiếp tục kiểm tra file đầu ra
        
        try:
            df_output = pd.read_csv(output_file, encoding=encoding, low_memory=False)
            logging.info(f"Đã đọc file đầu ra: {output_file}, kích thước: {df_output.shape}")
        except Exception as e:
            logging.error(f"Không thể đọc file đầu ra: {str(e)}")
            print(f"Lỗi: File đầu ra không hợp lệ. Chi tiết: {str(e)}")
            return False
        
        # Kiểm tra cơ bản về dữ liệu
        validation_results = []
        
        # Kiểm tra số lượng dòng
        if 'df_input' in locals():
            if df_output.shape[0] == 0:
                msg = "Dữ liệu đầu ra không có dòng nào"
                validation_results.append({"test": "Số lượng dòng", "result": "Thất bại", "message": msg})
                logging.error(msg)
            elif df_output.shape[0] > df_input.shape[0]:
                msg = f"Dữ liệu đầu ra có nhiều dòng hơn đầu vào ({df_output.shape[0]} > {df_input.shape[0]})"
                validation_results.append({"test": "Số lượng dòng", "result": "Cảnh báo", "message": msg})
                logging.warning(msg)
            else:
                removed_rows = df_input.shape[0] - df_output.shape[0]
                pct_removed = (removed_rows / df_input.shape[0]) * 100 if df_input.shape[0] > 0 else 0
                msg = f"Đã loại bỏ {removed_rows} dòng ({pct_removed:.1f}%)"
                validation_results.append({"test": "Số lượng dòng", "result": "Thành công", "message": msg})
                logging.info(msg)
        
        # Kiểm tra các cột
        required_columns = []  # Điền các cột bắt buộc vào đây nếu cần
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df_output.columns]
            if missing_columns:
                msg = f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}"
                validation_results.append({"test": "Cột bắt buộc", "result": "Thất bại", "message": msg})
                logging.error(msg)
            else:
                validation_results.append({"test": "Cột bắt buộc", "result": "Thành công", "message": "Tất cả cột bắt buộc đều có mặt"})
                
        # Kiểm tra giá trị null
        null_counts = df_output.isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            null_columns = null_counts[null_counts > 0].to_dict()
            null_msg = ", ".join([f"{col}: {count}" for col, count in null_columns.items()])
            msg = f"Dữ liệu vẫn còn {total_nulls} giá trị null ({null_msg})"
            validation_results.append({"test": "Giá trị null", "result": "Cảnh báo", "message": msg})
            logging.warning(msg)
        else:
            validation_results.append({"test": "Giá trị null", "result": "Thành công", "message": "Không có giá trị null"})
            logging.info("Không có giá trị null trong dữ liệu đầu ra")
        
        # Kiểm tra các dòng trùng lặp
        duplicates = df_output.duplicated().sum()
        if duplicates > 0:
            msg = f"Dữ liệu vẫn còn {duplicates} dòng trùng lặp"
            validation_results.append({"test": "Dòng trùng lặp", "result": "Cảnh báo", "message": msg})
            logging.warning(msg)
        else:
            validation_results.append({"test": "Dòng trùng lặp", "result": "Thành công", "message": "Không có dòng trùng lặp"})
            logging.info("Không có dòng trùng lặp trong dữ liệu đầu ra")
        
        # Kiểm tra định dạng cột nếu cần
        # (Thêm các kiểm tra tùy chỉnh khác nếu cần)
        
        # In kết quả kiểm tra
        print("\n=== KẾT QUẢ KIỂM TRA DỮ LIỆU ĐẦU RA ===")
        for result in validation_results:
            status_symbol = "✓" if result["result"] == "Thành công" else "⚠" if result["result"] == "Cảnh báo" else "✗"
            print(f"{status_symbol} {result['test']}: {result['message']}")
        
        # Tính toán điểm hợp lệ
        success_count = sum(1 for r in validation_results if r["result"] == "Thành công")
        warning_count = sum(1 for r in validation_results if r["result"] == "Cảnh báo")
        failure_count = sum(1 for r in validation_results if r["result"] == "Thất bại")
        
        total_tests = len(validation_results)
        if total_tests > 0:
            success_rate = (success_count / total_tests) * 100
            print(f"\nTỷ lệ thành công: {success_rate:.1f}% ({success_count}/{total_tests} kiểm tra)")
            print(f"Cảnh báo: {warning_count}, Lỗi: {failure_count}")
            
            if failure_count > 0:
                logging.error(f"Kiểm tra hợp lệ thất bại với {failure_count} lỗi nghiêm trọng")
                return False
            elif success_rate >= 80:  # Đạt ít nhất 80% kiểm tra thành công
                logging.info(f"Dữ liệu đầu ra hợp lệ với tỷ lệ thành công {success_rate:.1f}%")
                return True
            else:
                logging.warning(f"Dữ liệu đầu ra có thể không hợp lệ với tỷ lệ thành công thấp: {success_rate:.1f}%")
                return False
        else:
            logging.warning("Không có kiểm tra nào được thực hiện")
            return True  # Không có kiểm tra thì mặc định là hợp lệ
            
    except Exception as e:
        logging.exception(f"Lỗi khi kiểm tra tính hợp lệ của dữ liệu: {str(e)}")
        print(f"Lỗi khi kiểm tra tính hợp lệ của dữ liệu: {str(e)}")
        return False
    
def standardize_phone_numbers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Chuẩn hóa số điện thoại theo định dạng nhất quán.
    
    Hàm này loại bỏ các ký tự không phải số, thêm mã quốc gia nếu cần,
    và định dạng số điện thoại theo chuẩn thông dụng.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa cột cần chuẩn hóa
    column : str
        Tên cột chứa số điện thoại cần chuẩn hóa
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với cột số điện thoại đã được chuẩn hóa
    """
    if column not in df.columns:
        print(f"Cảnh báo: Cột '{column}' không tồn tại trong DataFrame")
        return df
    
    # Tạo bản sao để không thay đổi DataFrame gốc
    result_df = df.copy()
    
    # Bỏ qua các hàng có giá trị null
    mask = result_df[column].notna()
    original_values = result_df.loc[mask, column].astype(str)
    cleaned_values = original_values.copy()
    
    # Số lượng giá trị cần xử lý
    total_values = len(cleaned_values)
    if total_values == 0:
        print(f"Không có giá trị nào để chuẩn hóa trong cột '{column}'")
        return result_df
    
    print(f"Đang chuẩn hóa {total_values} số điện thoại trong cột '{column}'")
    
    # Loại bỏ tất cả ký tự không phải số
    cleaned_values = cleaned_values.str.replace(r'[^\d+]', '', regex=True)
    
    # Loại bỏ các số 0 ở đầu nếu không phải là mã quốc gia
    cleaned_values = cleaned_values.str.replace(r'^0+(?!\+)', '', regex=True)
    
    # Thêm mã quốc gia Việt Nam (+84) nếu chưa có và số có 9-10 chữ số
    # Định dạng: +84 xxx xxx xxx
    def format_vietnam_number(phone):
        if pd.isna(phone) or len(phone) == 0:
            return phone
        
        # Nếu đã có mã quốc tế bắt đầu bằng + hoặc 00, giữ nguyên
        if phone.startswith('+') or phone.startswith('00'):
            return phone
        
        # Nếu số bắt đầu bằng 84, thêm dấu +
        if phone.startswith('84') and len(phone) >= 11:
            return '+' + phone
        
        # Nếu số có 9-10 chữ số, thêm mã Việt Nam
        if 9 <= len(phone) <= 10:
            # Nếu bắt đầu bằng 0, loại bỏ số 0 đầu
            if phone.startswith('0'):
                phone = phone[1:]
            return '+84' + phone
        
        return phone
    
    cleaned_values = cleaned_values.apply(format_vietnam_number)
    
    # Định dạng lại số điện thoại để dễ đọc
    def format_phone_display(phone):
        if pd.isna(phone) or len(phone) == 0:
            return phone
        
        # Đối với số Việt Nam: +84 xxx xxx xxx
        if phone.startswith('+84') and len(phone) >= 10:
            national = phone[3:]  # Phần sau +84
            if len(national) == 9:  # 9 chữ số sau +84
                return f"+84 {national[:3]} {national[3:6]} {national[6:]}"
            elif len(national) == 10:  # 10 chữ số sau +84
                return f"+84 {national[:4]} {national[4:7]} {national[7:]}"
        
        return phone
    
    cleaned_values = cleaned_values.apply(format_phone_display)
    
    # Đếm số giá trị đã thay đổi
    changed_mask = original_values != cleaned_values
    num_changed = changed_mask.sum()
    
    # Cập nhật lại DataFrame
    result_df.loc[mask, column] = cleaned_values
    
    print(f"Đã chuẩn hóa {num_changed} số điện thoại trong cột '{column}'")
    
    return result_df

def standardize_addresses(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Chuẩn hóa địa chỉ theo định dạng nhất quán.
    
    Hàm này thực hiện các công việc như loại bỏ khoảng trắng thừa,
    chuẩn hóa từ viết tắt, định dạng lại mã bưu điện, v.v.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa cột cần chuẩn hóa
    column : str
        Tên cột chứa địa chỉ cần chuẩn hóa
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với cột địa chỉ đã được chuẩn hóa
    """
    if column not in df.columns:
        print(f"Cảnh báo: Cột '{column}' không tồn tại trong DataFrame")
        return df
    
    # Tạo bản sao để không thay đổi DataFrame gốc
    result_df = df.copy()
    
    # Bỏ qua các hàng có giá trị null
    mask = result_df[column].notna()
    original_values = result_df.loc[mask, column].astype(str)
    cleaned_values = original_values.copy()
    
    # Số lượng giá trị cần xử lý
    total_values = len(cleaned_values)
    if total_values == 0:
        print(f"Không có giá trị nào để chuẩn hóa trong cột '{column}'")
        return result_df
    
    print(f"Đang chuẩn hóa {total_values} địa chỉ trong cột '{column}'")
    
    # Loại bỏ khoảng trắng thừa
    cleaned_values = cleaned_values.str.strip()
    cleaned_values = cleaned_values.str.replace(r'\s+', ' ', regex=True)
    
    # Xử lý chữ hoa chữ thường cho địa chỉ
    # Title case cho địa chỉ, nhưng giữ nguyên các từ viết tắt
    def title_case_with_exceptions(address):
        if pd.isna(address) or not isinstance(address, str):
            return address
        
        # Danh sách các từ viết tắt cần giữ nguyên
        exceptions = ['TPHCM', 'TP.HCM', 'TP', 'HCM', 'ĐT', 'QL', 'P.', 'Q.', 'F.', 'D.', 'TT.']
        
        # Chuyển về title case
        titled = address.title()
        
        # Khôi phục các từ viết tắt
        for exception in exceptions:
            pattern = re.compile(r'\b' + re.escape(exception.title()) + r'\b')
            titled = pattern.sub(exception, titled)
        
        return titled
    
    cleaned_values = cleaned_values.apply(title_case_with_exceptions)
    
    # Chuẩn hóa các mẫu địa chỉ phổ biến của Việt Nam
    # Thay thế các từ viết tắt không nhất quán
    address_replacements = {
        r'\bDuong\b': 'Đường',
        r'\bPhuong\b': 'Phường',
        r'\bQuan\b': 'Quận',
        r'\bThanh Pho\b': 'Thành phố',
        r'\bTp\b': 'TP',
        r'\bTp\.\b': 'TP.',
        r'\bTp\. Hcm\b': 'TP.HCM',
        r'\bHo Chi Minh\b': 'Hồ Chí Minh',
        r'\bHcm\b': 'HCM',
        r'\bHa Noi\b': 'Hà Nội',
        r'\bSo\b': 'Số',
        r'\bTo\b': 'Tổ',
        r'\bKhu\b': 'Khu',
        r'\bKhu Pho\b': 'Khu phố',
        r'\bKp\b': 'KP',
        r'\bKp\.\b': 'KP.',
        r'\bThi Tran\b': 'Thị trấn',
        r'\bTt\b': 'TT',
        r'\bTt\.\b': 'TT.',
        r'\bXa\b': 'Xã',
        r'\bHuyen\b': 'Huyện',
        r'\bAp\b': 'Ấp',
        r'\bKhom\b': 'Khóm',
        r'\bThon\b': 'Thôn'
    }
    
    for pattern, replacement in address_replacements.items():
        cleaned_values = cleaned_values.str.replace(pattern, replacement, regex=True)
    
    # Chuẩn hóa dấu phẩy trong địa chỉ
    cleaned_values = cleaned_values.str.replace(r'\s*,\s*', ', ', regex=True)
    
    # Đếm số giá trị đã thay đổi
    changed_mask = original_values != cleaned_values
    num_changed = changed_mask.sum()
    
    # Cập nhật lại DataFrame
    result_df.loc[mask, column] = cleaned_values
    
    print(f"Đã chuẩn hóa {num_changed} địa chỉ trong cột '{column}'")
    
    return result_df

def handle_case_consistency(df: pd.DataFrame, columns: List[str] = None, case: str = 'lower') -> pd.DataFrame:
    """
    Xử lý tính nhất quán về chữ hoa/thường trong các cột văn bản.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    columns : List[str], optional
        Danh sách cột cần xử lý. Nếu None, xử lý tất cả cột kiểu object
    case : str, default 'lower'
        Loại biến đổi: 'lower' (chữ thường), 'upper' (chữ hoa), 'title' (viết hoa mỗi từ)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame đã xử lý
    """
    # Tạo bản sao để không thay đổi DataFrame gốc
    result_df = df.copy()
    
    # Nếu không chỉ định cột, lấy tất cả cột kiểu object
    if columns is None:
        columns = result_df.select_dtypes(include=['object']).columns.tolist()
    else:
        # Lọc ra các cột tồn tại và có kiểu object
        columns = [col for col in columns if col in result_df.columns and result_df[col].dtype == 'object']
    
    if not columns:
        print("Không có cột văn bản nào để xử lý tính nhất quán chữ hoa/thường")
        return result_df
    
    print(f"Đang xử lý tính nhất quán chữ hoa/thường cho {len(columns)} cột với phương pháp: {case}")
    
    for col in columns:
        # Bỏ qua giá trị null
        mask = result_df[col].notna()
        
        # Áp dụng phương pháp chuyển đổi
        if case == 'lower':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.lower()
        elif case == 'upper':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.upper()
        elif case == 'title':
            result_df.loc[mask, col] = result_df.loc[mask, col].str.title()
        
        print(f"   - Đã xử lý cột '{col}' với phương pháp {case}")
    
    return result_df

def process_columns_in_batches(
    df: pd.DataFrame, 
    process_func: callable, 
    columns: List[str] = None, 
    batch_size: int = 5000, 
    **kwargs
) -> pd.DataFrame:
    """
    Xử lý nhiều cột của DataFrame theo lô để tránh lỗi cache.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    process_func : callable
        Hàm xử lý áp dụng cho mỗi cột, nhận cột và trả về Series đã xử lý
    columns : List[str], optional
        Danh sách các cột cần xử lý. Nếu None, xử lý tất cả các cột
    batch_size : int, default 5000
        Kích thước của mỗi lô (số dòng)
    **kwargs : 
        Các tham số bổ sung cho hàm process_func
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột đã được xử lý
    """
    if df is None or df.empty:
        logging.warning("DataFrame trống hoặc None, không thể xử lý theo lô")
        return df
    
    # Xác định các cột cần xử lý
    if columns is None:
        columns = df.columns.tolist()
    else:
        # Chỉ giữ lại các cột tồn tại trong DataFrame
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        logging.warning("Không có cột nào để xử lý")
        return df
    
    # Nếu kích thước DataFrame <= batch_size, xử lý trực tiếp
    if len(df) <= batch_size or batch_size <= 0:
        result_df = df.copy()
        for col in columns:
            try:
                result_df[col] = process_func(df[col], **kwargs)
            except Exception as e:
                logging.error(f"Lỗi khi xử lý cột '{col}': {str(e)}")
        return result_df
    
    # Xử lý theo lô
    def _process_batch(batch_df):
        result_batch = batch_df.copy()
        for col in columns:
            try:
                result_batch[col] = process_func(batch_df[col], **kwargs)
            except Exception as e:
                logging.error(f"Lỗi khi xử lý cột '{col}' trong lô: {str(e)}")
        return result_batch
    
    # Sử dụng hàm xử lý theo lô
    return process_dataframe_in_batches(df, _process_batch, batch_size=batch_size)

def batch_convert_types(
    df: pd.DataFrame, 
    type_mapping: Dict[str, str], 
    batch_size: int = 5000
) -> pd.DataFrame:
    """
    Chuyển đổi kiểu dữ liệu của các cột theo lô.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    type_mapping : Dict[str, str]
        Ánh xạ tên cột với kiểu dữ liệu cần chuyển đổi
    batch_size : int, default 5000
        Kích thước của mỗi lô (số dòng)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột đã được chuyển đổi kiểu dữ liệu
    """
    if df is None or df.empty:
        logging.warning("DataFrame trống hoặc None, không thể xử lý")
        return df
    
    # Lọc ra các cột có trong DataFrame
    valid_mapping = {col: dtype for col, dtype in type_mapping.items() if col in df.columns}
    
    if not valid_mapping:
        logging.warning("Không có cột nào để chuyển đổi kiểu dữ liệu")
        return df
    
    # Hàm xử lý kiểu dữ liệu cho từng lô
    def _convert_types_batch(batch_df):
        result_df = batch_df.copy()
        
        for col, dtype in valid_mapping.items():
            try:
                if dtype == 'numeric':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                elif dtype == 'datetime':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                elif dtype == 'category':
                    result_df[col] = result_df[col].astype('category')
                elif dtype == 'string':
                    result_df[col] = result_df[col].astype(str)
                elif dtype == 'boolean':
                    result_df[col] = result_df[col].astype(bool)
                else:
                    result_df[col] = result_df[col].astype(dtype)
            except Exception as e:
                logging.error(f"Lỗi khi chuyển đổi cột '{col}' sang kiểu '{dtype}': {str(e)}")
        
        return result_df
    
    # Nếu DataFrame nhỏ hoặc batch_size <= 0, xử lý trực tiếp
    if len(df) <= batch_size or batch_size <= 0:
        return _convert_types_batch(df)
    
    # Sử dụng hàm xử lý theo lô
    return process_dataframe_in_batches(df, _convert_types_batch, batch_size=batch_size)
    
def auto_clean_data(df: pd.DataFrame, analysis_result: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Tự động làm sạch dữ liệu dựa trên kết quả phân tích.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần làm sạch
    analysis_result : dict
        Kết quả phân tích từ hàm analyze_and_suggest
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        DataFrame đã làm sạch và báo cáo kết quả
    """
    if df is None or df.empty:
        return df, {"error": "DataFrame trống"}
    
    # Tạo bản sao của DataFrame gốc để không thay đổi nó
    cleaned_df = df.copy()
    
    # Báo cáo kết quả làm sạch
    cleaning_report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "rows_after": None,
        "columns_after": None,
        "actions_taken": []
    }
    
    # Kiểm tra xem phân tích có tồn tại không
    if not analysis_result or "suggestions" not in analysis_result:
        return cleaned_df, {"error": "Kết quả phân tích không hợp lệ"}
    
    suggestions = analysis_result.get("suggestions", {})
    
    # 1. Xử lý dữ liệu trùng lặp
    duplicate_suggestions = suggestions.get("duplicate_data", [])
    if duplicate_suggestions:
        original_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        rows_removed = original_rows - len(cleaned_df)
        
        if rows_removed > 0:
            cleaning_report["actions_taken"].append({
                "action": "remove_duplicates",
                "details": f"Đã loại bỏ {rows_removed} dòng trùng lặp"
            })
    
    # 2. Xử lý outliers
    outlier_suggestions = suggestions.get("outliers", [])
    for outlier in outlier_suggestions:
        col = outlier.get("column")
        if col in cleaned_df.columns:
            # Áp dụng phương pháp IQR để xử lý outliers
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Thay thế outliers bằng giới hạn trên/dưới
            outlier_count = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            cleaning_report["actions_taken"].append({
                "action": "treat_outliers",
                "column": col,
                "details": f"Đã xử lý {outlier_count} outliers trong cột '{col}' bằng phương pháp IQR"
            })
    
    # 3. Xử lý giá trị thiếu
    missing_data_suggestions = suggestions.get("missing_data", [])
    for missing in missing_data_suggestions:
        col = missing.get("column")
        percent = missing.get("missing_percent", 0)
        suggestion = missing.get("suggestion", "")
        
        if col in cleaned_df.columns:
            # Nếu cột có quá nhiều giá trị thiếu (>80%), có thể loại bỏ
            if percent > 80 and "loại bỏ cột" in suggestion.lower():
                cleaned_df = cleaned_df.drop(columns=[col])
                cleaning_report["actions_taken"].append({
                    "action": "drop_column",
                    "column": col,
                    "details": f"Đã loại bỏ cột '{col}' vì có {percent:.1f}% giá trị thiếu"
                })
            # Nếu cột có nhiều giá trị thiếu (>50% nhưng <=80%), điền giá trị thiếu
            elif percent > 50:
                # Điền bằng giá trị phổ biến nhất cho cột không phải số
                if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    most_common = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "N/A"
                    cleaned_df[col] = cleaned_df[col].fillna(most_common)
                    cleaning_report["actions_taken"].append({
                        "action": "fill_missing",
                        "column": col,
                        "method": "mode",
                        "details": f"Đã điền {percent:.1f}% giá trị thiếu trong cột '{col}' bằng giá trị phổ biến nhất"
                    })
                # Điền bằng median cho cột số
                else:
                    median_value = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_value)
                    cleaning_report["actions_taken"].append({
                        "action": "fill_missing",
                        "column": col,
                        "method": "median",
                        "details": f"Đã điền {percent:.1f}% giá trị thiếu trong cột '{col}' bằng giá trị trung vị"
                    })
            # Nếu cột có ít giá trị thiếu (<=50%)
            else:
                # Điền giá trị thiếu dựa vào kiểu dữ liệu
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Dùng mean cho cột số liên tục
                    if cleaned_df[col].nunique() > 10:
                        mean_value = cleaned_df[col].mean()
                        cleaned_df[col] = cleaned_df[col].fillna(mean_value)
                        cleaning_report["actions_taken"].append({
                            "action": "fill_missing",
                            "column": col,
                            "method": "mean",
                            "details": f"Đã điền {percent:.1f}% giá trị thiếu trong cột '{col}' bằng giá trị trung bình"
                        })
                    # Dùng median cho cột số không liên tục
                    else:
                        median_value = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(median_value)
                        cleaning_report["actions_taken"].append({
                            "action": "fill_missing",
                            "column": col,
                            "method": "median",
                            "details": f"Đã điền {percent:.1f}% giá trị thiếu trong cột '{col}' bằng giá trị trung vị"
                        })
                else:
                    # Dùng mode cho cột không phải số
                    most_common = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "N/A"
                    cleaned_df[col] = cleaned_df[col].fillna(most_common)
                    cleaning_report["actions_taken"].append({
                        "action": "fill_missing",
                        "column": col,
                        "method": "mode",
                        "details": f"Đã điền {percent:.1f}% giá trị thiếu trong cột '{col}' bằng giá trị phổ biến nhất"
                    })
    
    # 4. Chuẩn hóa văn bản
    text_data_suggestions = suggestions.get("text_data", [])
    for text_suggestion in text_data_suggestions:
        col = text_suggestion.get("column")
        issue = text_suggestion.get("issue", "")
        
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            # Loại bỏ khoảng trắng thừa
            if "khoảng trắng thừa" in issue:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaning_report["actions_taken"].append({
                    "action": "normalize_text",
                    "column": col,
                    "method": "strip",
                    "details": f"Đã loại bỏ khoảng trắng thừa trong cột '{col}'"
                })
            
            # Chuẩn hóa chữ hoa/thường
            if "không nhất quán chữ hoa/thường" in issue:
                suggested_case = text_suggestion.get("suggestion", "").split("sang dạng ")[-1] if "sang dạng " in text_suggestion.get("suggestion", "") else ""
                
                if suggested_case == "lower":
                    cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
                    method = "chữ thường"
                elif suggested_case == "upper":
                    cleaned_df[col] = cleaned_df[col].astype(str).str.upper()
                    method = "chữ hoa"
                elif suggested_case == "title":
                    cleaned_df[col] = cleaned_df[col].astype(str).str.title()
                    method = "viết hoa đầu từ"
                
                cleaning_report["actions_taken"].append({
                    "action": "normalize_text",
                    "column": col,
                    "method": suggested_case,
                    "details": f"Đã chuẩn hóa cột '{col}' sang dạng {method}"
                })
    
    # 5. Chuyển đổi kiểu dữ liệu
    data_types_suggestions = suggestions.get("data_types", [])
    for type_suggestion in data_types_suggestions:
        col = type_suggestion.get("column")
        suggested_type = type_suggestion.get("suggested_type", "")
        
        if col in cleaned_df.columns:
            # Chuyển đổi thành kiểu số
            if suggested_type == "numeric":
                original_type = cleaned_df[col].dtype
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Điền giá trị thiếu sau khi chuyển đổi
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                
                cleaning_report["actions_taken"].append({
                    "action": "convert_type",
                    "column": col,
                    "from_type": str(original_type),
                    "to_type": str(cleaned_df[col].dtype),
                    "details": f"Đã chuyển đổi cột '{col}' từ kiểu {original_type} sang kiểu số"
                })
    
    # Chuyển đổi dữ liệu ngày tháng
    date_data_suggestions = suggestions.get("date_data", [])
    for date_suggestion in date_data_suggestions:
        col = date_suggestion.get("column")
        
        if col in cleaned_df.columns:
            try:
                original_type = cleaned_df[col].dtype
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                
                cleaning_report["actions_taken"].append({
                    "action": "convert_type",
                    "column": col,
                    "from_type": str(original_type),
                    "to_type": "datetime64[ns]",
                    "details": f"Đã chuyển đổi cột '{col}' sang kiểu ngày tháng"
                })
            except:
                # Nếu không thể chuyển đổi, giữ nguyên
                pass
    
    # Cập nhật thông tin sau khi làm sạch
    cleaning_report["rows_after"] = len(cleaned_df)
    cleaning_report["columns_after"] = len(cleaned_df.columns)
    cleaning_report["rows_removed"] = cleaning_report["rows_before"] - cleaning_report["rows_after"]
    cleaning_report["columns_removed"] = cleaning_report["columns_before"] - cleaning_report["columns_after"]
    
    # Tóm tắt kết quả
    print("\n===== KẾT QUẢ LÀM SẠCH DỮ LIỆU =====")
    print(f"Số dòng: {cleaning_report['rows_before']:,} -> {cleaning_report['rows_after']:,} ({cleaning_report['rows_removed']} dòng đã bị loại bỏ)")
    print(f"Số cột: {cleaning_report['columns_before']} -> {cleaning_report['columns_after']} ({cleaning_report['columns_removed']} cột đã bị loại bỏ)")
    print("\nCác hành động đã thực hiện:")
    for action in cleaning_report["actions_taken"]:
        print(f"  - {action['details']}")
    
    return cleaned_df, cleaning_report
    
def apply_suggested_formats(df: pd.DataFrame, suggestions: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Áp dụng các định dạng được gợi ý cho DataFrame để đảm bảo tính nhất quán.
    Nếu không cung cấp suggestions, sẽ tự động phân tích và áp dụng định dạng tốt nhất.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần áp dụng định dạng
    suggestions : Dict[str, Dict[str, Any]], optional
        Từ điển chứa thông tin gợi ý định dạng (từ hàm suggest_column_formats)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi áp dụng định dạng
    """
    from src.data_validator import suggest_column_formats
    
    # Tạo một bản sao để không làm thay đổi DataFrame gốc
    result_df = df.copy()
    
    # Nếu không có gợi ý định dạng, tự động phân tích
    if suggestions is None:
        print("Phân tích và gợi ý định dạng tự động...")
        suggestions = suggest_column_formats(df)
    
    # Áp dụng các chuyển đổi kiểu dữ liệu
    conversions_applied = 0
    
    for col, suggestion in suggestions.items():
        if col not in result_df.columns:
            continue
            
        current_type = str(result_df[col].dtype)
        suggested_type = suggestion.get('suggested_type')
        
        if suggested_type and current_type != suggested_type:
            try:
                # Xử lý các trường hợp đặc biệt
                if suggested_type == 'datetime64[ns]' or suggested_type == 'datetime64':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                elif suggested_type == 'int64' or suggested_type == 'Int64':
                    # Kiểm tra xem cột có giá trị NA không
                    if result_df[col].isna().any():
                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('Int64')
                    else:
                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('int64')
                elif suggested_type == 'float64':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('float64')
                elif suggested_type == 'category':
                    result_df[col] = result_df[col].astype('category')
                elif suggested_type == 'string':
                    result_df[col] = result_df[col].astype('string')
                elif suggested_type == 'bool':
                    result_df[col] = result_df[col].astype('bool')
                else:
                    # Thử chuyển đổi kiểu dữ liệu trực tiếp
                    result_df[col] = result_df[col].astype(suggested_type)
                
                conversions_applied += 1
                print(f"Đã chuyển cột '{col}' từ {current_type} sang {suggested_type}")
            except Exception as e:
                print(f"Không thể chuyển cột '{col}' sang kiểu {suggested_type}: {str(e)}")
    
    # Thêm metadata về định dạng vào DataFrame để sử dụng khi xuất
    format_metadata = {}
    for col, suggestion in suggestions.items():
        if col in result_df.columns and 'suggested_format' in suggestion:
            format_metadata[col] = suggestion['suggested_format']
    
    # Lưu metadata định dạng vào thuộc tính của DataFrame
    result_df.attrs['format_metadata'] = format_metadata
    
    print(f"Đã áp dụng {conversions_applied} chuyển đổi kiểu dữ liệu")
    print(f"Đã lưu metadata định dạng cho {len(format_metadata)} cột")
    
    return result_df

def optimize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối ưu hóa kiểu dữ liệu cho các cột trong DataFrame
    để giảm mức sử dụng bộ nhớ và cải thiện hiệu suất.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần tối ưu hóa
    
    Returns:
    --------
    pd.DataFrame
        DataFrame đã được tối ưu hóa kiểu dữ liệu
    """
    result_df = df.copy()
    
    # Theo dõi mức sử dụng bộ nhớ trước khi tối ưu hóa
    memory_usage_before = result_df.memory_usage(deep=True).sum()
    
    # Xử lý các cột số
    for col in result_df.select_dtypes(include=['int', 'float']).columns:
        # Với số nguyên, chuyển sang kiểu nhỏ nhất có thể
        if pd.api.types.is_integer_dtype(result_df[col]):
            # Kiểm tra có giá trị null không
            has_null = result_df[col].isna().any()
            
            if not has_null:
                # Tìm khoảng giá trị
                col_min, col_max = result_df[col].min(), result_df[col].max()
                
                # Chọn kiểu dữ liệu phù hợp
                if col_min >= 0:
                    if col_max < 2**8:
                        result_df[col] = result_df[col].astype('uint8')
                    elif col_max < 2**16:
                        result_df[col] = result_df[col].astype('uint16')
                    elif col_max < 2**32:
                        result_df[col] = result_df[col].astype('uint32')
                    else:
                        result_df[col] = result_df[col].astype('uint64')
                else:
                    if col_min > -2**7 and col_max < 2**7:
                        result_df[col] = result_df[col].astype('int8')
                    elif col_min > -2**15 and col_max < 2**15:
                        result_df[col] = result_df[col].astype('int16')
                    elif col_min > -2**31 and col_max < 2**31:
                        result_df[col] = result_df[col].astype('int32')
                    else:
                        result_df[col] = result_df[col].astype('int64')
            else:
                # Với cột có null, sử dụng kiểu Int64
                result_df[col] = result_df[col].astype('Int64')
        
        # Với số thực, kiểm tra nếu có thể dùng float32
        elif pd.api.types.is_float_dtype(result_df[col]):
            # Thử float32 nếu không mất mát độ chính xác quá nhiều
            try:
                float32_col = result_df[col].astype('float32')
                precision_loss = (abs(result_df[col] - float32_col) > 1e-6).mean()
                
                # Nếu mất mát độ chính xác < 0.1%, dùng float32
                if precision_loss < 0.001:
                    result_df[col] = float32_col
            except:
                pass
    
    # Xử lý các cột văn bản
    for col in result_df.select_dtypes(include=['object']).columns:
        # Kiểm tra nếu cột này có thể là categorical
        nunique = result_df[col].nunique()
        if nunique < len(result_df) * 0.5:  # Nếu số giá trị duy nhất < 50% số dòng
            result_df[col] = result_df[col].astype('category')
    
    # Tính toán mức sử dụng bộ nhớ sau khi tối ưu
    memory_usage_after = result_df.memory_usage(deep=True).sum()
    memory_saved = memory_usage_before - memory_usage_after
    memory_saved_mb = memory_saved / (1024 * 1024)
    
    print(f"Đã tối ưu hóa kiểu dữ liệu cho {len(df.columns)} cột")
    print(f"Mức sử dụng bộ nhớ: {memory_usage_before / (1024 * 1024):.2f} MB -> {memory_usage_after / (1024 * 1024):.2f} MB")
    print(f"Tiết kiệm: {memory_saved_mb:.2f} MB ({memory_saved / memory_usage_before * 100:.1f}%)")
    
    return result_df
    