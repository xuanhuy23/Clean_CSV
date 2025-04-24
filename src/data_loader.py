import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dateutil.parser import parse
import json
import logging

def load_data(
    filepath,
    sep=',',
    encoding='utf-8',
    na_values=None,
    use_metadata=True,
    **kwargs
):
    """
    Đọc dữ liệu từ file với cơ chế tự động phát hiện định dạng.
    Hỗ trợ CSV, Excel và các định dạng khác.
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file dữ liệu
    sep : str, default ','
        Ký tự phân cách trong file CSV
    encoding : str, default 'utf-8'
        Mã hóa ký tự cho file CSV
    na_values : list, default None
        Các giá trị được coi là NA/NaN
    use_metadata : bool, default True
        Nếu True, kiểm tra và sử dụng file metadata (nếu có) để khôi phục kiểu dữ liệu
    **kwargs : dict
        Các tham số bổ sung cho hàm đọc dữ liệu
        
    Returns:
    --------
    tuple
        DataFrame đã đọc và thông tin bổ sung
    """
    # Kiểm tra file có tồn tại
    if not os.path.exists(filepath):
        print(f"Lỗi: File '{filepath}' không tồn tại")
        logging.error(f"File '{filepath}' không tồn tại")
        return None, {'error': f"File '{filepath}' không tồn tại"}
    
    # Phát hiện định dạng file
    file_extension = os.path.splitext(filepath)[1].lower()
    
    try:
        # Kiểm tra kích thước file
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"Cảnh báo: File '{filepath}' trống (0 byte)")
            logging.warning(f"File '{filepath}' trống (0 byte)")
            return pd.DataFrame(), {'warning': 'File trống'}
        
        # Hiển thị thông tin file
        print(f"Đang đọc file: {filepath} ({file_size / 1024:.1f} KB)")
        
        # Đọc file dựa trên định dạng
        if file_extension in ['.xlsx', '.xls']:
            print(f"Đã phát hiện file Excel ({file_extension})")
            try:
                df = pd.read_excel(filepath, **kwargs)
                load_info = {'format': 'excel', 'sheet_name': kwargs.get('sheet_name', 0)}
            except Exception as e:
                print(f"Lỗi khi đọc file Excel: {str(e)}")
                logging.error(f"Lỗi khi đọc file Excel '{filepath}': {str(e)}")
                return None, {'error': str(e)}
                
        elif file_extension == '.csv':
            print(f"Đã phát hiện file CSV (separator: '{sep}', encoding: {encoding})")
            try:
                # Đọc dữ liệu cơ bản từ CSV
                df = pd.read_csv(filepath, sep=sep, encoding=encoding, na_values=na_values, **kwargs)
                load_info = {'format': 'csv', 'encoding': encoding, 'separator': sep}
                
                # Kiểm tra xem có file metadata đi kèm không
                if use_metadata:
                    metadata_path = f"{os.path.splitext(filepath)[0]}.meta.json"
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                
                            print(f"Đã tìm thấy file metadata: {metadata_path}")
                            logging.info(f"Đang áp dụng metadata từ: {metadata_path}")
                            
                            # Áp dụng kiểu dữ liệu từ metadata
                            if 'column_types' in metadata and 'format_hints' in metadata:
                                # Tạo từ điển chuyển đổi kiểu dữ liệu
                                dtype_conversions = {}
                                date_columns = []
                                
                                for col, format_hint in metadata['format_hints'].items():
                                    if col in df.columns:
                                        if format_hint == 'integer':
                                            # Nếu có giá trị NA, phải dùng 'Int64' chứ không phải 'int64'
                                            if df[col].isna().any():
                                                dtype_conversions[col] = 'Int64'
                                            else:
                                                dtype_conversions[col] = 'int64'
                                        elif format_hint.startswith('float:'):
                                            dtype_conversions[col] = 'float64'
                                        elif format_hint == 'datetime':
                                            date_columns.append(col)
                                        elif format_hint == 'category':
                                            dtype_conversions[col] = 'category'
                                
                                # Áp dụng chuyển đổi kiểu dữ liệu
                                for col, dtype in dtype_conversions.items():
                                    try:
                                        df[col] = df[col].astype(dtype)
                                    except Exception as e:
                                        logging.warning(f"Không thể chuyển cột '{col}' sang kiểu {dtype}: {str(e)}")
                                
                                # Chuyển đổi các cột ngày tháng
                                for col in date_columns:
                                    try:
                                        df[col] = pd.to_datetime(df[col], errors='coerce')
                                    except Exception as e:
                                        logging.warning(f"Không thể chuyển cột '{col}' sang kiểu datetime: {str(e)}")
                                
                                print(f"Đã áp dụng kiểu dữ liệu từ metadata cho {len(dtype_conversions) + len(date_columns)} cột")
                                logging.info(f"Đã áp dụng kiểu dữ liệu từ metadata cho {len(dtype_conversions) + len(date_columns)} cột")
                        except Exception as e:
                            logging.warning(f"Không thể sử dụng metadata: {str(e)}")
                            print(f"Cảnh báo: Không thể sử dụng metadata: {str(e)}")
                
            except UnicodeDecodeError:
                print(f"Lỗi mã hóa ký tự. Thử với encoding khác...")
                alt_encodings = ['latin1', 'utf-16', 'ISO-8859-1', 'cp1252']
                for alt_encoding in alt_encodings:
                    if alt_encoding != encoding:
                        try:
                            print(f"  -> Thử với encoding: {alt_encoding}")
                            df = pd.read_csv(filepath, sep=sep, encoding=alt_encoding, na_values=na_values, **kwargs)
                            load_info = {'format': 'csv', 'encoding': alt_encoding, 'separator': sep}
                            print(f"  -> Đọc thành công với encoding: {alt_encoding}")
                            logging.info(f"Đọc thành công file CSV với encoding thay thế: {alt_encoding}")
                            break
                        except:
                            continue
                else:
                    print("Không thể đọc file với các encoding đã thử. Vui lòng kiểm tra lại file.")
                    logging.error(f"Không thể đọc file CSV '{filepath}' với các encoding đã thử")
                    return None, {'error': 'Lỗi encoding'}
            except Exception as e:
                print(f"Lỗi khi đọc file CSV: {str(e)}")
                logging.error(f"Lỗi khi đọc file CSV '{filepath}': {str(e)}")
                return None, {'error': str(e)}
        else:
            print(f"Định dạng file không được hỗ trợ: {file_extension}")
            logging.error(f"Định dạng file không được hỗ trợ: {file_extension}")
            return None, {'error': f"Định dạng file không được hỗ trợ: {file_extension}"}
    
        # Kiểm tra dataframe
        if df is None:
            print("Lỗi: Không đọc được dữ liệu từ file")
            logging.error(f"Không đọc được dữ liệu từ file '{filepath}'")
            return None, {'error': 'Lỗi không xác định khi đọc file'}
            
        if df.empty:
            print("Cảnh báo: DataFrame trống")
            logging.warning(f"DataFrame trống từ file '{filepath}'")
            return df, {'warning': 'DataFrame trống'}
        
        # Thêm thông tin về DataFrame
        load_info['rows'] = df.shape[0]
        load_info['columns'] = df.shape[1]
        load_info['memory_usage'] = df.memory_usage(deep=True).sum()
        
        logging.info(f"Đã đọc thành công file: {filepath} ({df.shape[0]} dòng x {df.shape[1]} cột)")
        return df, load_info
    
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file: {str(e)}")
        logging.exception(f"Lỗi không xác định khi đọc file '{filepath}'")
        return None, {'error': str(e)}

def load_csv(
    filepath,
    sep=',',
    encoding='utf-8',
    header=0,
    names=None,
    na_values=None,
    on_bad_lines='warn',
    usecols=None,
    dtype=None,
    parse_dates=None,
    skiprows=None,
    nrows=None,
    skipfooter=0,
    index_col=None
):
    """
    Tải dữ liệu từ file CSV vào DataFrame Pandas
    
    Tham số:
    - filepath (str): Đường dẫn đến file CSV
    - sep (str): Ký tự phân cách giữa các cột, mặc định là dấu phẩy
    - encoding (str): Quy tắc mã hóa ký tự của file
    - header (int/list): Dòng được sử dụng làm tên cột
    - names (list): Danh sách tên cột tùy chỉnh
    - na_values (list/dict): Các chuỗi bổ sung cần nhận diện là NaN
    - on_bad_lines (str): Hành động khi gặp dòng có số trường không đúng
    - usecols (list): Chỉ đọc các cột được chỉ định
    - dtype (dict): Chỉ định trước kiểu dữ liệu cho các cột
    - parse_dates (list/dict): Cố gắng phân tích cú pháp các cột thành datetime
    - skiprows (int/list): Bỏ qua các dòng ở đầu
    - nrows (int): Chỉ đọc số lượng dòng nhất định
    - skipfooter (int): Bỏ qua các dòng ở cuối
    - index_col (int/str): Đặt cột làm chỉ mục của DataFrame
    
    Trả về:
    - DataFrame: Dữ liệu đã tải từ file CSV
    """
    try:
        # Xác định các giá trị thiếu bổ sung (ngoài mặc định)
        default_na_values = ['NA', 'N/A', 'NULL', 'NaN', 'nan', '', 'None', 'UNKNOWN', 'ERROR']
        if na_values:
            if isinstance(na_values, list):
                na_values.extend(default_na_values)
            else:
                na_values = default_na_values
        else:
            na_values = default_na_values
            
        # Tải dữ liệu từ CSV
        df = pd.read_csv(
            filepath,
            sep=sep,
            encoding=encoding,
            header=header,
            names=names,
            na_values=na_values,
            on_bad_lines=on_bad_lines,
            usecols=usecols,
            dtype=dtype,
            parse_dates=parse_dates,
            skiprows=skiprows,
            nrows=nrows,
            skipfooter=skipfooter,
            index_col=index_col
        )
        
        print(f"Đã tải thành công dữ liệu từ: {filepath}")
        return df
    
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {str(e)}")
        return None

def explore_data(df):
    """
    Khám phá sơ bộ dữ liệu trong DataFrame
    
    Tham số:
    - df (DataFrame): DataFrame cần khám phá
    
    Trả về:
    - dict: Từ điển chứa các kết quả khám phá dữ liệu
    """
    if df is None or df.empty:
        print("DataFrame trống hoặc không tồn tại.")
        return None
    
    exploration_results = {}
    
    # Kích thước dữ liệu
    exploration_results['shape'] = df.shape
    print(f"Kích thước dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")
    
    # Xem nhanh dữ liệu
    exploration_results['head'] = df.head()
    print("\nNăm dòng đầu tiên:")
    print(df.head())
    
    exploration_results['tail'] = df.tail()
    print("\nNăm dòng cuối cùng:")
    print(df.tail())
    
    exploration_results['sample'] = df.sample(5 if df.shape[0] >= 5 else df.shape[0])
    print("\nNăm dòng ngẫu nhiên:")
    print(exploration_results['sample'])
    
    # Thông tin cấu trúc dữ liệu
    print("\nThông tin cấu trúc dữ liệu:")
    exploration_results['info'] = df.info()
    
    # Thống kê mô tả
    exploration_results['describe'] = df.describe(include='all')
    print("\nThống kê mô tả:")
    print(df.describe(include='all'))
    
    # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()
    exploration_results['missing_values'] = missing_values
    print("\nGiá trị thiếu trong từng cột:")
    print(missing_values)
    
    # Kiểm tra kiểu dữ liệu
    exploration_results['dtypes'] = df.dtypes
    print("\nKiểu dữ liệu của từng cột:")
    print(df.dtypes)
    
    # Kiểm tra giá trị duy nhất trong mỗi cột
    unique_values = {}
    for col in df.columns:
        # Nếu số lượng giá trị duy nhất ít, liệt kê chúng
        if df[col].nunique() < 20:
            unique_values[col] = df[col].value_counts().to_dict()
    
    exploration_results['unique_values'] = unique_values
    print("\nGiá trị duy nhất trong các cột có ít giá trị khác nhau:")
    for col, values in unique_values.items():
        print(f"{col}: {values}")
    
    return exploration_results

def detect_date_format(value):
    """
    Phát hiện định dạng ngày tháng từ một chuỗi
    
    Tham số:
    - value (str): Chuỗi cần phát hiện định dạng ngày tháng
    
    Trả về:
    - str: Định dạng ngày tháng nếu phát hiện được, None nếu không phát hiện được
    """
    if not isinstance(value, str):
        return None
        
    formats = [
        # Định dạng MM/DD/YYYY hoặc DD/MM/YYYY
        (r'\d{1,2}/\d{1,2}/\d{4}', '%m/%d/%Y'),
        (r'\d{1,2}/\d{1,2}/\d{2}', '%m/%d/%y'),
        # Định dạng YYYY-MM-DD
        (r'\d{4}-\d{1,2}-\d{1,2}', '%Y-%m-%d'),
        # Định dạng YYYY/MM/DD
        (r'\d{4}/\d{1,2}/\d{1,2}', '%Y/%m/%d'),
        # Định dạng DD-MM-YYYY
        (r'\d{1,2}-\d{1,2}-\d{4}', '%d-%m-%Y'),
        # Định dạng DD.MM.YYYY
        (r'\d{1,2}\.\d{1,2}\.\d{4}', '%d.%m.%Y'),
        # Định dạng YYYY.MM.DD
        (r'\d{4}\.\d{1,2}\.\d{1,2}', '%Y.%m.%d'),
        # Định dạng MMM DD, YYYY (Jan 01, 2020)
        (r'[A-Za-z]{3}\s+\d{1,2},\s+\d{4}', '%b %d, %Y'),
        # Định dạng DD MMM YYYY (01 Jan 2020)
        (r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}', '%d %b %Y'),
        # Định dạng YYYY-MM-DD HH:MM:SS
        (r'\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}', '%Y-%m-%d %H:%M:%S'),
        # Định dạng DD/MM/YYYY HH:MM:SS
        (r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}', '%d/%m/%Y %H:%M:%S'),
        # Định dạng MM/DD/YYYY HH:MM:SS
        (r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}', '%m/%d/%Y %H:%M:%S'),
    ]
    
    for pattern, date_format in formats:
        if re.match(pattern, value):
            try:
                # Thử chuyển đổi với định dạng đã phát hiện
                datetime.strptime(value, date_format)
                return date_format
            except ValueError:
                # Nếu chuyển đổi thất bại, thử định dạng khác
                continue
    
    return None

def detect_common_date_format(df: pd.DataFrame, column: str) -> Optional[str]:
    """
    Phát hiện định dạng ngày tháng phổ biến nhất trong một cột
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    column : str
        Tên cột cần phát hiện định dạng
        
    Returns:
    --------
    str hoặc None
        Định dạng ngày tháng phổ biến nhất trong cột, hoặc None nếu không phát hiện được
    """
    if column not in df.columns:
        print(f"Cảnh báo: Cột '{column}' không tồn tại trong DataFrame")
        return None
        
    # Lấy các giá trị không null
    values = df[column].dropna().astype(str).tolist()
    if not values:
        return None
        
    # Giới hạn số lượng mẫu để phân tích
    sample_size = min(1000, len(values))
    sample_values = values[:sample_size]
    
    # Các định dạng ngày tháng phổ biến để thử
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
        '%b %d, %Y', '%d %b %Y', '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M',
        '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M', '%m-%d-%Y %H:%M'
    ]
    
    # Đếm số lần mỗi định dạng phù hợp
    format_counts = Counter()
    
    for value in sample_values:
        for date_format in date_formats:
            try:
                pd.to_datetime(value, format=date_format)
                format_counts[date_format] += 1
                # Chỉ cần tìm được một định dạng phù hợp cho giá trị này
                break
            except:
                continue
    
    if not format_counts:
        return None
        
    # Tìm định dạng phổ biến nhất
    most_common_format, count = format_counts.most_common(1)[0]
    
    # Kiểm tra định dạng phổ biến nhất có phù hợp với ít nhất 50% giá trị không
    if count / len(sample_values) >= 0.5:
        print(f"Đã phát hiện định dạng ngày tháng '{most_common_format}' cho {count}/{len(sample_values)} giá trị")
        return most_common_format
    else:
        print(f"Không tìm thấy định dạng ngày tháng phổ biến cho cột '{column}'")
        return None

def detect_data_issues(df: pd.DataFrame) -> Dict:
    """
    Phát hiện các vấn đề cơ bản trong dữ liệu
    
    Tham số:
    - df (DataFrame): DataFrame cần kiểm tra
    
    Trả về:
    - dict: Từ điển chứa các vấn đề phát hiện được
    """
    issues = {}
    
    # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()
    issues['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # Kiểm tra giá trị trùng lặp
    issues['duplicate_rows'] = df.duplicated().sum()
    
    # Kiểm tra sự không nhất quán trong các cột chuỗi
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    inconsistent_strings = {}
    
    for col in string_columns:
        # Tính số lượng các giá trị có khoảng trắng thừa
        if df[col].dtype == 'object':
            values_with_spaces = df[col].str.strip() != df[col]
            if values_with_spaces.any():
                inconsistent_strings[col] = values_with_spaces.sum()
    
    issues['inconsistent_strings'] = inconsistent_strings
    
    # Kiểm tra cột có thể là số nhưng được lưu dưới dạng chuỗi
    potential_numeric = {}
    
    for col in string_columns:
        # Thử chuyển đổi sang số
        try:
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            if not numeric_conversion.isna().all() and numeric_conversion.notna().any():
                # Nếu ít nhất một giá trị có thể chuyển đổi, nhưng không phải tất cả
                potential_numeric[col] = numeric_conversion.notna().sum()
        except:
            pass
    
    issues['potential_numeric'] = potential_numeric
    
    # Kiểm tra cột có thể là ngày tháng nhưng được lưu dưới dạng chuỗi
    potential_dates = {}
    date_formats = {}
    
    for col in string_columns:
        # Bỏ qua cột rỗng hoặc có quá nhiều giá trị null
        if df[col].isna().sum() > 0.9 * len(df):
            continue
            
        # Bỏ qua các cột có tên chỉ rõ là không phải ngày tháng (ví dụ: _name, _id, station, address, ...)
        if any(keyword in col.lower() for keyword in ['name', 'station', 'address', 'street', 'city', 'state', 'zip']):
            continue
            
        try:
            # Lấy mẫu nhỏ để kiểm tra
            sample_size = min(100, len(df[col].dropna()))
            sample = df[col].dropna().astype(str).head(sample_size).tolist()
            
            # Kiểm tra xem mẫu có dạng văn bản dài (> 20 ký tự) không phải ngày tháng
            if any(len(str(x)) > 20 for x in sample) or any(str(x).count(' ') > 2 for x in sample):
                continue
                
            # Phát hiện định dạng ngày tháng phổ biến trong cột
            date_format = detect_common_date_format(df, col)
            
            if date_format:
                # Sử dụng định dạng đã phát hiện để chuyển đổi
                # Thử chuyển đổi với một số lượng mẫu trước
                success_count = 0
                for x in sample:
                    try:
                        if pd.notna(x):
                            datetime.strptime(str(x).strip(), date_format)
                            success_count += 1
                    except (ValueError, TypeError):
                        pass
                
                # Nếu ít nhất 50% mẫu chuyển đổi thành công
                if success_count >= 0.5 * len(sample):
                    from dateutil import parser
                    
                    def parse_with_format(x, format_str):
                        try:
                            if pd.isna(x):
                                return pd.NaT
                            return datetime.strptime(str(x).strip(), format_str)
                        except (ValueError, TypeError):
                            return pd.NaT
                    
                    date_conversion = df[col].apply(lambda x: parse_with_format(x, date_format))
                    valid_conversions = (~pd.isna(date_conversion)).sum()
                    
                    if valid_conversions > 0.5 * len(df[col].dropna()):  # Nếu trên 50% giá trị hợp lệ
                        # Lưu số lượng giá trị có thể chuyển đổi và định dạng đã phát hiện
                        potential_dates[col] = valid_conversions
                        date_formats[col] = date_format
                        print(f"Đã phát hiện định dạng '{date_format}' cho cột '{col}' với {valid_conversions} giá trị hợp lệ")
            else:
                # Nếu không phát hiện được mẫu, sử dụng chuyển đổi thử nghiệm với mẫu nhỏ
                # Thử chuyển đổi một mẫu nhỏ các giá trị thay vì toàn bộ cột
                try:
                    # Chỉ thử với mẫu
                    date_sample = pd.to_datetime(sample, errors='coerce')
                    valid_count = date_sample.notna().sum()
                    
                    # Nếu trên 70% mẫu chuyển đổi thành công
                    if valid_count > 0.7 * len(sample):
                        # Thử chuyển đổi thêm một mẫu lớn hơn để xác nhận
                        larger_sample_size = min(1000, len(df[col].dropna()))
                        larger_sample = df[col].dropna().astype(str).head(larger_sample_size).tolist()
                        larger_date_sample = pd.to_datetime(larger_sample, errors='coerce')
                        larger_valid_count = larger_date_sample.notna().sum()
                        
                        if larger_valid_count > 0.7 * len(larger_sample):
                            potential_dates[col] = larger_valid_count
                            print(f"Cột '{col}' có thể chứa ngày tháng, với {larger_valid_count}/{len(larger_sample)} giá trị hợp lệ")
                except Exception as e:
                    # Bỏ qua lỗi khi chuyển đổi
                    pass
        except Exception as e:
            print(f"Lỗi khi kiểm tra cột '{col}' là ngày tháng: {str(e)}")
    
    issues['potential_dates'] = potential_dates
    issues['date_formats'] = date_formats
    
    # In kết quả
    print("PHÁT HIỆN VẤN ĐỀ DỮ LIỆU:")
    
    print(f"\n1. Số lượng dòng trùng lặp: {issues['duplicate_rows']}")
    
    print("\n2. Số lượng giá trị thiếu trong từng cột:")
    for col, count in issues['missing_values'].items():
        print(f"   - {col}: {count} giá trị")
    
    print("\n3. Các cột chuỗi có khoảng trắng thừa:")
    for col, count in issues['inconsistent_strings'].items():
        print(f"   - {col}: {count} giá trị")
    
    print("\n4. Các cột có thể là số nhưng được lưu dưới dạng chuỗi:")
    for col, count in issues['potential_numeric'].items():
        print(f"   - {col}: {count} giá trị có thể chuyển đổi")
    
    print("\n5. Các cột có thể là ngày tháng nhưng được lưu dưới dạng chuỗi:")
    for col, count in issues['potential_dates'].items():
        print(f"   - {col}: {count} giá trị có thể chuyển đổi")
        if col in date_formats:
            print(f"      Định dạng phát hiện: {date_formats[col]}")
        
    return issues

def analyze_and_suggest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phân tích dữ liệu và đưa ra các gợi ý làm sạch dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần phân tích
        
    Returns:
    --------
    Dict[str, Any]
        Từ điển chứa kết quả phân tích và gợi ý
    """
    if df is None or df.empty:
        print("Không thể phân tích: DataFrame trống hoặc None")
        return {"error": "DataFrame trống hoặc None"}
    
    suggestions = {
        "missing_data": [],
        "duplicate_data": [],
        "data_types": [],
        "outliers": [],
        "text_data": [],
        "date_data": []
    }
    
    # Thông tin cơ bản về DataFrame
    row_count = df.shape[0]
    col_count = df.shape[1]
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    # Phân tích giá trị thiếu
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / row_count) * 100
    problematic_missing = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    # Đưa ra gợi ý về các cột có quá nhiều giá trị thiếu
    for col, percent in problematic_missing.items():
        if percent > 80:
            suggestions["missing_data"].append({
                "column": col,
                "missing_percent": percent,
                "suggestion": "Cân nhắc loại bỏ cột này vì quá nhiều giá trị thiếu"
            })
        elif percent > 50:
            suggestions["missing_data"].append({
                "column": col,
                "missing_percent": percent,
                "suggestion": "Điền giá trị thiếu hoặc loại bỏ cột"
            })
        elif percent > 20:
            # Kiểm tra kiểu dữ liệu để đề xuất phương pháp phù hợp
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions["missing_data"].append({
                    "column": col, 
                    "missing_percent": percent,
                    "suggestion": "Điền bằng median hoặc mean"
                })
            else:
                suggestions["missing_data"].append({
                    "column": col, 
                    "missing_percent": percent,
                    "suggestion": "Điền bằng mode hoặc giá trị phổ biến nhất"
                })
        else:
            suggestions["missing_data"].append({
                "column": col, 
                "missing_percent": percent,
                "suggestion": "Điền giá trị thiếu hoặc loại bỏ dòng"
            })
    
    # Kiểm tra dữ liệu trùng lặp
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        dup_percent = (dup_rows / row_count) * 100
        suggestions["duplicate_data"].append({
            "duplicate_count": dup_rows,
            "duplicate_percent": dup_percent,
            "suggestion": "Loại bỏ các dòng trùng lặp"
        })
    
    # Phân tích kiểu dữ liệu và đề xuất chuyển đổi
    for col in df.columns:
        # Kiểm tra cột số có chứa dưới dạng chuỗi
        if df[col].dtype == 'object':
            # Thử chuyển sang số
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            if numeric_conversion.notna().sum() > 0.8 * df[col].count():
                suggestions["data_types"].append({
                    "column": col,
                    "current_type": str(df[col].dtype),
                    "suggested_type": "numeric",
                    "suggestion": "Chuyển đổi thành kiểu số"
                })
                continue
                
            # Kiểm tra cột ngày tháng
            date_count = 0
            sample = df[col].dropna().head(10).astype(str)
            date_patterns = [
                r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}',  # YYYY-MM-DD
                r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}',  # DD-MM-YYYY
                r'\d{1,2}[-/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/\.]\d{4}',  # DD-Mon-YYYY
            ]
            
            for val in sample:
                if any(re.search(pattern, val) for pattern in date_patterns):
                    date_count += 1
            
            if date_count >= 0.7 * len(sample):
                suggestions["date_data"].append({
                    "column": col,
                    "suggestion": "Chuyển đổi thành kiểu ngày tháng"
                })
                continue
                
            # Kiểm tra cột văn bản cần chuẩn hóa
            if df[col].str.strip().ne(df[col]).any():
                suggestions["text_data"].append({
                    "column": col,
                    "issue": "khoảng trắng thừa",
                    "suggestion": "Chuẩn hóa văn bản"
                })
            
            # Kiểm tra tính nhất quán chữ hoa/thường
            if not df[col].str.islower().all() and not df[col].str.isupper().all() and not df[col].str.istitle().all():
                case_lower = df[col].str.islower().sum()
                case_upper = df[col].str.isupper().sum()
                case_title = df[col].str.istitle().sum()
                
                max_case = max(case_lower, case_upper, case_title)
                suggested_case = "lower" if max_case == case_lower else "upper" if max_case == case_upper else "title"
                
                suggestions["text_data"].append({
                    "column": col,
                    "issue": "không nhất quán chữ hoa/thường",
                    "suggestion": f"Chuyển đổi sang dạng {suggested_case}"
                })
        
        # Kiểm tra outliers cho cột số
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Bỏ qua các cột Boolean hoặc tất cả giá trị như nhau
            if df[col].nunique() <= 2 or df[col].std() == 0:
                continue
                
            # Kiểm tra outliers bằng phương pháp IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:  # Tránh chia cho 0
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percent = (outliers / df[col].count()) * 100
                
                if outlier_percent > 5:
                    suggestions["outliers"].append({
                        "column": col,
                        "outlier_count": outliers,
                        "outlier_percent": outlier_percent,
                        "suggestion": "Xử lý outliers bằng phương pháp IQR"
                    })
    
    # Tạo tóm tắt phân tích
    summary = {
        "row_count": row_count,
        "column_count": col_count,
        "memory_usage_mb": memory_usage,
        "missing_columns": len(problematic_missing),
        "duplicate_rows": dup_rows,
        "suggested_actions": {
            "missing_data": len(suggestions["missing_data"]),
            "duplicate_data": len(suggestions["duplicate_data"]),
            "data_types": len(suggestions["data_types"]),
            "outliers": len(suggestions["outliers"]),
            "text_data": len(suggestions["text_data"]),
            "date_data": len(suggestions["date_data"])
        }
    }
    
    # Tạo kế hoạch làm sạch tự động
    cleaning_plan = []
    
    # Ưu tiên xử lý dữ liệu trùng lặp
    if suggestions["duplicate_data"]:
        cleaning_plan.append("1. Loại bỏ các dòng trùng lặp")
    
    # Xử lý outliers
    if suggestions["outliers"]:
        cleaning_plan.append(f"2. Xử lý {len(suggestions['outliers'])} cột có outliers")
    
    # Xử lý giá trị thiếu
    if suggestions["missing_data"]:
        cleaning_plan.append(f"3. Xử lý {len(suggestions['missing_data'])} cột có giá trị thiếu")
    
    # Chuẩn hóa văn bản
    if suggestions["text_data"]:
        cleaning_plan.append(f"4. Chuẩn hóa {len(suggestions['text_data'])} cột văn bản")
    
    # Chuyển đổi kiểu dữ liệu
    if suggestions["data_types"] or suggestions["date_data"]:
        cleaning_plan.append(f"5. Chuyển đổi kiểu dữ liệu cho {len(suggestions['data_types']) + len(suggestions['date_data'])} cột")
    
    # In kết quả phân tích
    print("\n===== PHÂN TÍCH VÀ GỢI Ý LÀM SẠCH DỮ LIỆU =====")
    print(f"DataFrame: {row_count:,} dòng x {col_count} cột")
    print(f"Dung lượng bộ nhớ: {memory_usage:.2f} MB")
    print(f"Số cột có giá trị thiếu: {len(problematic_missing)}")
    print(f"Số dòng trùng lặp: {dup_rows} ({(dup_rows / row_count * 100):.2f}% tổng số dòng)" if dup_rows > 0 else "Không có dòng trùng lặp")
    
    print("\nKẾ HOẠCH LÀM SẠCH DỮ LIỆU GỢI Ý:")
    for step in cleaning_plan:
        print(f"  {step}")
    
    return {
        "summary": summary,
        "suggestions": suggestions,
        "cleaning_plan": cleaning_plan
    }
