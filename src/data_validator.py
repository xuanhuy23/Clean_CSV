import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

def validate_cleaned_data(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                          key_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Xác thực dữ liệu sau khi làm sạch bằng cách so sánh với dữ liệu gốc
    và thực hiện các kiểm tra chất lượng.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        DataFrame gốc trước khi làm sạch
    df_cleaned : pd.DataFrame
        DataFrame đã làm sạch
    key_cols : List[str], optional
        Danh sách các cột quan trọng cần kiểm tra kỹ hơn
        
    Returns:
    --------
    Dict[str, Any]
        Từ điển chứa kết quả xác thực
    """
    validation_results = {}
    
    # Kiểm tra kích thước
    validation_results['shape'] = {
        'original': df_original.shape,
        'cleaned': df_cleaned.shape,
        'difference': (df_original.shape[0] - df_cleaned.shape[0], 
                      df_original.shape[1] - df_cleaned.shape[1])
    }
    
    # Kiểm tra kiểu dữ liệu
    validation_results['dtypes'] = {
        'original': df_original.dtypes.to_dict(),
        'cleaned': df_cleaned.dtypes.to_dict()
    }
    
    # Kiểm tra giá trị thiếu
    validation_results['missing_values'] = {
        'original': df_original.isnull().sum().to_dict(),
        'cleaned': df_cleaned.isnull().sum().to_dict(),
        'original_total': df_original.isnull().sum().sum(),
        'cleaned_total': df_cleaned.isnull().sum().sum()
    }
    
    # Thống kê mô tả cho các cột số
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        validation_results['numeric_stats'] = {}
        for col in numeric_cols:
            if col in df_original.columns:
                original_stats = df_original[col].describe().to_dict()
                cleaned_stats = df_cleaned[col].describe().to_dict()
                validation_results['numeric_stats'][col] = {
                    'original': original_stats,
                    'cleaned': cleaned_stats
                }
    
    # Kiểm tra giá trị duy nhất cho các cột danh mục
    if key_cols:
        validation_results['unique_values'] = {}
        for col in key_cols:
            if col in df_cleaned.columns and col in df_original.columns:
                original_unique = df_original[col].nunique()
                cleaned_unique = df_cleaned[col].nunique()
                validation_results['unique_values'][col] = {
                    'original_count': original_unique,
                    'cleaned_count': cleaned_unique,
                    'difference': original_unique - cleaned_unique
                }
                
                # Nếu số lượng giá trị duy nhất ít, lưu lại các giá trị cụ thể
                if cleaned_unique < 20:
                    validation_results['unique_values'][col]['cleaned_values'] = df_cleaned[col].value_counts().to_dict()
    
    # Xuất kết quả thành báo cáo
    print("\n===== XÁC THỰC DỮ LIỆU SAU LÀM SẠCH =====")
    
    print(f"\n1. Kiểm tra kích thước:")
    print(f"   - DataFrame gốc: {validation_results['shape']['original'][0]} dòng x {validation_results['shape']['original'][1]} cột")
    print(f"   - DataFrame đã làm sạch: {validation_results['shape']['cleaned'][0]} dòng x {validation_results['shape']['cleaned'][1]} cột")
    print(f"   - Thay đổi: {validation_results['shape']['difference'][0]} dòng, {validation_results['shape']['difference'][1]} cột")
    
    print("\n2. Kiểm tra giá trị thiếu:")
    print(f"   - Tổng số giá trị thiếu ban đầu: {validation_results['missing_values']['original_total']}")
    print(f"   - Tổng số giá trị thiếu sau làm sạch: {validation_results['missing_values']['cleaned_total']}")
    print(f"   - Tỷ lệ giá trị đầy đủ sau làm sạch: {100 - (validation_results['missing_values']['cleaned_total'] * 100 / (df_cleaned.shape[0] * df_cleaned.shape[1])):.2f}%")
    
    print("\n3. Kiểm tra kiểu dữ liệu:")
    print("   - Kiểu dữ liệu sau làm sạch:")
    for col, dtype in validation_results['dtypes']['cleaned'].items():
        original_dtype = validation_results['dtypes']['original'].get(col, 'N/A')
        if str(original_dtype) != str(dtype):
            print(f"     + {col}: {original_dtype} -> {dtype} (đã thay đổi)")
        else:
            print(f"     + {col}: {dtype}")
    
    if 'numeric_stats' in validation_results:
        print("\n4. Thống kê dữ liệu số:")
        for col, stats in validation_results['numeric_stats'].items():
            print(f"   - Cột '{col}':")
            if 'min' in stats['cleaned'] and 'max' in stats['cleaned']:
                print(f"     + Khoảng giá trị: [{stats['cleaned']['min']:.2f}, {stats['cleaned']['max']:.2f}]")
            if 'mean' in stats['cleaned'] and 'std' in stats['cleaned']:
                print(f"     + Trung bình: {stats['cleaned']['mean']:.2f} (±{stats['cleaned']['std']:.2f})")
            if '25%' in stats['cleaned'] and '50%' in stats['cleaned'] and '75%' in stats['cleaned']:
                print(f"     + Phân vị: Q1={stats['cleaned']['25%']:.2f}, Q2={stats['cleaned']['50%']:.2f}, Q3={stats['cleaned']['75%']:.2f}")
    
    if 'unique_values' in validation_results:
        print("\n5. Kiểm tra giá trị duy nhất cho các cột quan trọng:")
        for col, unique_info in validation_results['unique_values'].items():
            print(f"   - Cột '{col}':")
            print(f"     + Số lượng giá trị duy nhất ban đầu: {unique_info['original_count']}")
            print(f"     + Số lượng giá trị duy nhất sau làm sạch: {unique_info['cleaned_count']}")
            if unique_info['difference'] != 0:
                print(f"     + Đã giảm {unique_info['difference']} giá trị duy nhất")
            
            if 'cleaned_values' in unique_info:
                print(f"     + Các giá trị duy nhất và số lần xuất hiện:")
                for value, count in unique_info['cleaned_values'].items():
                    print(f"       * {value}: {count}")
    
    return validation_results

def validate_column_distribution(df: pd.DataFrame, col: str, 
                                lower_bound: Optional[float] = None,
                                upper_bound: Optional[float] = None) -> Dict[str, Any]:
    """
    Kiểm tra phân phối của một cột cụ thể trong DataFrame.
    Hữu ích để kiểm tra xem cột có nằm trong khoảng giá trị hợp lý sau khi xử lý outlier không.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần kiểm tra
    col : str
        Tên cột cần kiểm tra phân phối
    lower_bound : float, optional
        Ngưỡng dưới của dữ liệu
    upper_bound : float, optional
        Ngưỡng trên của dữ liệu
    
    Returns:
    --------
    Dict[str, Any]
        Thông tin về phân phối của cột
    """
    if col not in df.columns:
        print(f"Cột '{col}' không tồn tại trong DataFrame")
        return {}
    
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Cột '{col}' không phải là kiểu số")
        return {}
    
    # Lấy thống kê cơ bản
    stats = df[col].describe()
    
    # Kiểm tra giá trị nằm ngoài khoảng
    out_of_range = {}
    if lower_bound is not None:
        below_min = (df[col] < lower_bound).sum()
        out_of_range['below_min'] = below_min
        out_of_range['below_min_percent'] = (below_min / len(df)) * 100
    
    if upper_bound is not None:
        above_max = (df[col] > upper_bound).sum()
        out_of_range['above_max'] = above_max
        out_of_range['above_max_percent'] = (above_max / len(df)) * 100
    
    result = {
        'stats': stats.to_dict(),
        'out_of_range': out_of_range
    }
    
    # In kết quả
    print(f"\n===== KIỂM TRA PHÂN PHỐI CỘT '{col}' =====")
    print(f"- Số lượng giá trị: {stats['count']}")
    print(f"- Giá trị nhỏ nhất: {stats['min']:.2f}")
    print(f"- Giá trị lớn nhất: {stats['max']:.2f}")
    print(f"- Trung bình: {stats['mean']:.2f}")
    print(f"- Độ lệch chuẩn: {stats['std']:.2f}")
    print(f"- Phân vị: Q1={stats['25%']:.2f}, Q2={stats['50%']:.2f}, Q3={stats['75%']:.2f}")
    
    if out_of_range:
        print("\nKiểm tra giá trị nằm ngoài khoảng:")
        if 'below_min' in out_of_range:
            print(f"- Số giá trị < {lower_bound}: {out_of_range['below_min']} ({out_of_range['below_min_percent']:.2f}%)")
        if 'above_max' in out_of_range:
            print(f"- Số giá trị > {upper_bound}: {out_of_range['above_max']} ({out_of_range['above_max_percent']:.2f}%)")
    
    return result

def validate_data_consistency(df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kiểm tra tính nhất quán của dữ liệu dựa trên các quy tắc cụ thể
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần kiểm tra
    rules : Dict[str, Any]
        Các quy tắc kiểm tra, ví dụ:
        {
            'range_check': {'column': 'age', 'min': 0, 'max': 120},
            'uniqueness': {'columns': ['id']},
            'relationship': {'columns': ['start_date', 'end_date'], 'condition': 'start_date <= end_date'}
        }
    
    Returns:
    --------
    Dict[str, Any]
        Kết quả kiểm tra tính nhất quán
    """
    results = {}
    
    for rule_name, rule_params in rules.items():
        if rule_name == 'range_check':
            col = rule_params.get('column')
            min_val = rule_params.get('min')
            max_val = rule_params.get('max')
            
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum() if min_val is not None and max_val is not None else 0
                results[f'{col}_range_check'] = {
                    'total': len(df),
                    'out_of_range': out_of_range,
                    'out_of_range_percent': (out_of_range / len(df)) * 100
                }
        
        elif rule_name == 'uniqueness':
            cols = rule_params.get('columns', [])
            if all(col in df.columns for col in cols):
                duplicates = df.duplicated(subset=cols).sum()
                results[f"{'_'.join(cols)}_uniqueness"] = {
                    'total': len(df),
                    'duplicates': duplicates,
                    'duplicates_percent': (duplicates / len(df)) * 100
                }
    
    # In kết quả
    print("\n===== KIỂM TRA TÍNH NHẤT QUÁN DỮ LIỆU =====")
    for check_name, check_results in results.items():
        print(f"- {check_name}:")
        for key, value in check_results.items():
            if isinstance(value, float):
                print(f"  + {key}: {value:.2f}")
            else:
                print(f"  + {key}: {value}")
    
    return results

def validate_data_formats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Xác thực định dạng dữ liệu của các cột trong DataFrame.
    Kiểm tra xem dữ liệu có phù hợp với kiểu dữ liệu đang được sử dụng không.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xác thực định dạng
        
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Từ điển chứa thông tin xác thực định dạng cho từng cột
    """
    result = {}
    
    for col in df.columns:
        col_result = {
            'type': str(df[col].dtype),
            'issues': [],
            'suggestion': None,
            'format_consistent': True
        }
        
        # Kiểm tra xem cột có rỗng không
        if df[col].isna().all():
            col_result['issues'].append('Cột hoàn toàn rỗng')
            col_result['format_consistent'] = False
            result[col] = col_result
            continue
        
        # Lấy mẫu giá trị không NA
        non_na_values = df[col].dropna()
        if len(non_na_values) == 0:
            col_result['issues'].append('Cột không có giá trị nào')
            col_result['format_consistent'] = False
            result[col] = col_result
            continue
        
        # Kiểm tra tính nhất quán của định dạng dựa trên kiểu dữ liệu
        if pd.api.types.is_numeric_dtype(df[col]):
            # Kiểm tra nếu là kiểu số
            if pd.api.types.is_integer_dtype(df[col]):
                # Với số nguyên, kiểm tra xem có phải tất cả đều là số nguyên
                numeric_check = non_na_values.apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()))
                if not numeric_check.all():
                    col_result['issues'].append('Cột kiểu số nguyên nhưng có giá trị không phải số nguyên')
                    col_result['format_consistent'] = False
                    col_result['suggestion'] = 'float64'
            else:
                # Với số thực, kiểm tra xem có thể chuyển đổi sang số không
                try:
                    # Kiểm tra xem có bao nhiêu chữ số thập phân
                    decimal_places = non_na_values.astype(str).str.extract(r'\.(\d+)')[0].str.len().value_counts().index
                    if len(decimal_places) > 1:
                        col_result['issues'].append(f'Cột kiểu số thực nhưng có số chữ số thập phân không nhất quán')
                        col_result['decimal_places_found'] = list(decimal_places)
                        col_result['format_consistent'] = False
                except:
                    pass
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Kiểm tra định dạng ngày tháng
            try:
                # Chuyển đổi datetimes sang chuỗi để kiểm tra định dạng
                date_strings = non_na_values.dt.strftime('%Y-%m-%d %H:%M:%S').astype(str)
                # Kiểm tra xem có bao nhiêu kiểu có giờ phút giây và bao nhiêu chỉ có ngày 
                with_time = date_strings.str.contains(' 00:00:00$', regex=True).sum()
                only_date = len(date_strings) - with_time
                
                if with_time > 0 and only_date > 0:
                    col_result['issues'].append('Cột ngày tháng có cả giá trị với giờ phút giây và không có giờ phút giây')
                    col_result['format_consistent'] = False
                    col_result['suggestion'] = 'Chuẩn hóa thành định dạng ngày hoặc định dạng ngày giờ'
            except:
                col_result['issues'].append('Cột kiểu ngày tháng không thể kiểm tra định dạng')
                col_result['format_consistent'] = False
        
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Kiểm tra cột chuỗi xem có thể chuyển sang kiểu khác không
            
            # Kiểm tra nếu tất cả là số
            if pd.to_numeric(non_na_values, errors='coerce').notna().all():
                if pd.to_numeric(non_na_values, errors='coerce').apply(lambda x: x.is_integer() if isinstance(x, float) else True).all():
                    col_result['issues'].append('Cột kiểu chuỗi nhưng chứa toàn bộ giá trị số nguyên')
                    col_result['format_consistent'] = False
                    col_result['suggestion'] = 'int64'
                else:
                    col_result['issues'].append('Cột kiểu chuỗi nhưng chứa toàn bộ giá trị số thực')
                    col_result['format_consistent'] = False
                    col_result['suggestion'] = 'float64'
            
            # Kiểm tra nếu tất cả là ngày tháng
            if pd.to_datetime(non_na_values, errors='coerce').notna().all():
                col_result['issues'].append('Cột kiểu chuỗi nhưng chứa toàn bộ giá trị ngày tháng')
                col_result['format_consistent'] = False
                col_result['suggestion'] = 'datetime64'
            
            # Kiểm tra độ dài chuỗi có nhất quán không
            string_lengths = non_na_values.astype(str).str.len().value_counts()
            if len(string_lengths) > 5:  # Nếu có quá nhiều độ dài khác nhau
                col_result['issues'].append(f'Cột kiểu chuỗi có {len(string_lengths)} độ dài khác nhau')
                col_result['format_consistent'] = False
        
        elif pd.api.types.is_categorical_dtype(df[col]):
            # Với categorical, kiểm tra số lượng giá trị duy nhất
            n_unique = df[col].nunique()
            if n_unique > 100 and n_unique > 0.5 * len(df):
                col_result['issues'].append(f'Cột kiểu phân loại nhưng có quá nhiều giá trị duy nhất ({n_unique})')
                col_result['format_consistent'] = False
                col_result['suggestion'] = 'string'
        
        # Lưu kết quả kiểm tra
        result[col] = col_result
    
    # In kết quả kiểm tra
    print("\n===== KIỂM TRA ĐỊNH DẠNG DỮ LIỆU =====")
    
    consistent_count = sum(1 for col_result in result.values() if col_result['format_consistent'])
    inconsistent_count = len(result) - consistent_count
    
    print(f"- Tổng số cột: {len(result)}")
    print(f"- Số cột có định dạng nhất quán: {consistent_count}")
    print(f"- Số cột có vấn đề định dạng: {inconsistent_count}")
    
    if inconsistent_count > 0:
        print("\nDanh sách cột có vấn đề định dạng:")
        for col, col_result in result.items():
            if not col_result['format_consistent']:
                print(f"  + {col} (kiểu: {col_result['type']}):")
                for issue in col_result['issues']:
                    print(f"     - {issue}")
                if col_result['suggestion']:
                    print(f"     - Đề xuất: Chuyển sang kiểu {col_result['suggestion']}")
    
    return result

def suggest_column_formats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Gợi ý định dạng tốt nhất cho từng cột dựa trên phân tích dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần phân tích
    
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Từ điển chứa thông tin gợi ý định dạng cho từng cột
    """
    suggestions = {}
    
    for col in df.columns:
        suggestion = {
            'current_type': str(df[col].dtype),
            'suggested_type': None,
            'suggested_format': None,
            'reason': None
        }
        
        # Bỏ qua cột toàn giá trị NA
        if df[col].isna().all():
            suggestion['suggested_type'] = 'object'
            suggestion['reason'] = 'Cột toàn giá trị NA'
            suggestions[col] = suggestion
            continue
        
        non_na_values = df[col].dropna()
        
        # Kiểm tra nếu đang là kiểu số
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                # Kiểm tra nếu toàn bộ là số nguyên
                suggestion['suggested_type'] = 'int64'
                suggestion['suggested_format'] = '#,##0'  # Định dạng số nguyên với phân tách hàng nghìn
                suggestion['reason'] = 'Giữ nguyên kiểu số nguyên'
            else:
                # Với số thực, phát hiện số chữ số thập phân
                try:
                    decimal_places = non_na_values.astype(str).str.extract(r'\.(\d+)')[0].str.len().value_counts()
                    if not decimal_places.empty:
                        most_common_decimal = decimal_places.index[0]
                        suggestion['suggested_type'] = 'float64'
                        suggestion['suggested_format'] = f'#,##0.{"0" * most_common_decimal}'  # Định dạng với số chữ số thập phân phù hợp
                        suggestion['reason'] = f'Số thực với {most_common_decimal} chữ số thập phân'
                    else:
                        suggestion['suggested_type'] = 'float64'
                        suggestion['suggested_format'] = '#,##0.00'  # Mặc định 2 chữ số thập phân
                        suggestion['reason'] = 'Số thực mặc định 2 chữ số thập phân'
                except:
                    suggestion['suggested_type'] = 'float64'
                    suggestion['suggested_format'] = '#,##0.00'
                    suggestion['reason'] = 'Số thực không thể xác định số chữ số thập phân'
        
        # Kiểm tra nếu đang là datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Kiểm tra xem có giá trị giờ không
            has_time_info = False
            try:
                # Lấy mẫu để kiểm tra
                sample_values = non_na_values.head(100)
                # Kiểm tra xem có giá trị giờ khác 0 không
                has_time = (sample_values.dt.hour != 0).any() or (sample_values.dt.minute != 0).any() or (sample_values.dt.second != 0).any()
                
                if has_time:
                    suggestion['suggested_type'] = 'datetime64[ns]'
                    suggestion['suggested_format'] = 'DD/MM/YYYY HH:MM:SS'
                    suggestion['reason'] = 'Ngày tháng có thông tin giờ'
                else:
                    suggestion['suggested_type'] = 'datetime64[ns]'
                    suggestion['suggested_format'] = 'DD/MM/YYYY'
                    suggestion['reason'] = 'Ngày tháng không có thông tin giờ'
            except:
                suggestion['suggested_type'] = 'datetime64[ns]'
                suggestion['suggested_format'] = 'DD/MM/YYYY'
                suggestion['reason'] = 'Ngày tháng không thể kiểm tra thông tin giờ'
        
        # Kiểm tra nếu đang là object hoặc string
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Kiểm tra xem có thể chuyển thành số không
            numeric_conversion = pd.to_numeric(non_na_values, errors='coerce')
            if numeric_conversion.notna().all():
                # Nếu tất cả là số nguyên
                if numeric_conversion.apply(lambda x: x.is_integer() if isinstance(x, float) else True).all():
                    suggestion['suggested_type'] = 'int64'
                    suggestion['suggested_format'] = '#,##0'
                    suggestion['reason'] = 'Chuỗi có thể chuyển thành số nguyên'
                else:
                    suggestion['suggested_type'] = 'float64'
                    suggestion['suggested_format'] = '#,##0.00'
                    suggestion['reason'] = 'Chuỗi có thể chuyển thành số thực'
            
            # Kiểm tra xem có thể chuyển thành datetime không
            elif pd.to_datetime(non_na_values, errors='coerce').notna().all():
                suggestion['suggested_type'] = 'datetime64[ns]'
                suggestion['suggested_format'] = 'DD/MM/YYYY'
                suggestion['reason'] = 'Chuỗi có thể chuyển thành ngày tháng'
            
            # Kiểm tra số giá trị duy nhất để xem có nên chuyển thành categorical không
            elif df[col].nunique() < len(df) * 0.1:  # Ít hơn 10% là giá trị duy nhất
                suggestion['suggested_type'] = 'category'
                suggestion['suggested_format'] = '@'  # Định dạng văn bản
                suggestion['reason'] = f'Chuỗi có thể chuyển thành phân loại ({df[col].nunique()} giá trị duy nhất)'
            else:
                suggestion['suggested_type'] = 'string'
                suggestion['suggested_format'] = '@'  # Định dạng văn bản
                suggestion['reason'] = 'Giữ nguyên kiểu chuỗi'
        
        # Kiểm tra nếu đang là categorical
        elif pd.api.types.is_categorical_dtype(df[col]):
            # Kiểm tra xem có quá nhiều giá trị duy nhất không
            n_unique = df[col].nunique()
            if n_unique > 100 and n_unique > 0.5 * len(df):
                suggestion['suggested_type'] = 'string'
                suggestion['suggested_format'] = '@'
                suggestion['reason'] = f'Phân loại có quá nhiều giá trị duy nhất ({n_unique}), nên chuyển sang chuỗi'
            else:
                suggestion['suggested_type'] = 'category'
                suggestion['suggested_format'] = '@'
                suggestion['reason'] = 'Giữ nguyên kiểu phân loại'
        
        # Kiểm tra nếu đang là boolean
        elif pd.api.types.is_bool_dtype(df[col]):
            suggestion['suggested_type'] = 'bool'
            suggestion['suggested_format'] = 'TRUE/FALSE'
            suggestion['reason'] = 'Giữ nguyên kiểu boolean'
        
        # Các kiểu khác
        else:
            suggestion['suggested_type'] = str(df[col].dtype)
            suggestion['suggested_format'] = '@'
            suggestion['reason'] = 'Không có gợi ý định dạng'
        
        suggestions[col] = suggestion
    
    # In kết quả gợi ý
    print("\n===== GỢI Ý ĐỊNH DẠNG DỮ LIỆU =====")
    print(f"- Tổng số cột: {len(df.columns)}")
    
    # Chia theo loại gợi ý
    type_groups = {}
    for col, suggestion in suggestions.items():
        suggested_type = suggestion['suggested_type']
        if suggested_type not in type_groups:
            type_groups[suggested_type] = []
        type_groups[suggested_type].append(col)
    
    # In thông tin nhóm theo kiểu dữ liệu
    for dtype, cols in type_groups.items():
        print(f"\n  + Kiểu {dtype}: {len(cols)} cột")
        for col in cols[:5]:  # Chỉ hiển thị 5 cột đầu tiên để tránh quá dài
            suggestion = suggestions[col]
            print(f"     - {col}: {suggestion['current_type']} -> {suggestion['suggested_type']} ({suggestion['suggested_format']})")
        if len(cols) > 5:
            print(f"     - ... và {len(cols) - 5} cột khác")
    
    return suggestions
