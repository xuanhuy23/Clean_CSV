#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# Thêm thư mục gốc vào sys.path để import các module trong src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, detect_data_issues, analyze_and_suggest
from src.data_cleaner import (
    delete_missing_values, fill_missing_values, normalize_text, standardize_dates,
    convert_data_types, remove_duplicates, handle_outliers, convert_to_numeric, convert_to_category,
    standardize_phone_numbers, standardize_addresses, handle_case_consistency, auto_clean_data,
    optimize_datatypes, apply_suggested_formats
)
from src.data_validator import validate_cleaned_data, validate_column_distribution, validate_data_formats, suggest_column_formats
from src.utils import save_data, get_summary_stats, generate_html_report, simplify_html_report, export_json_report, process_dataframe_in_batches, export_to_excel

def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Thiết lập cấu hình logging
    
    Parameters:
    -----------
    log_dir : str, default "logs"
        Thư mục lưu file log
    log_level : str, default "INFO"
        Mức độ chi tiết của log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Chuyển đổi chuỗi log level thành hằng số logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Tạo thư mục log nếu chưa tồn tại
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Cảnh báo: Không thể tạo thư mục log: {str(e)}")
        log_dir = os.getcwd()
    
    # Tạo tên file log với timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_cleaning_{timestamp}.log")
    
    # Cấu hình logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Bắt đầu quá trình làm sạch dữ liệu. File log: {log_file}")
    return log_file

def parse_arguments() -> argparse.Namespace:
    """
    Phân tích tham số dòng lệnh
    
    Returns:
    --------
    argparse.Namespace
        Các tham số dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Công cụ làm sạch dữ liệu CSV')
    
    # Tham số đầu vào/đầu ra cơ bản
    parser.add_argument('--input', type=str, default='data/raw/input.csv',
                        help='Đường dẫn đến file CSV đầu vào')
    
    parser.add_argument('--output', type=str, default='data/processed/output.csv',
                        help='Đường dẫn đến file đầu ra (CSV hoặc Excel)')
    
    parser.add_argument('--encoding', type=str, default='utf-8',
                        help='Mã hóa ký tự của file đầu vào và đầu ra')
    
    parser.add_argument('--sep', type=str, default=',',
                        help='Ký tự phân cách cột (mặc định là dấu phẩy)')
    
    parser.add_argument('--excel', action='store_true', default=False,
                        help='Xuất dữ liệu đã làm sạch sang định dạng Excel (.xlsx)')
    
    parser.add_argument('--sheet-name', type=str, default='Data',
                        help='Tên sheet trong file Excel (chỉ áp dụng với --excel)')
    
    parser.add_argument('--na-values', type=str, nargs='+', 
                        default=['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN', 'None', 'UNKNOWN'],
                        help='Các giá trị được coi là NA')
    
    # Tùy chọn làm sạch tất cả (mặc định là True)
    parser.add_argument('--clean-all', action='store_true', default=True,
                        help='Thực hiện tất cả các tùy chọn làm sạch (mặc định bật)')
    
    parser.add_argument('--no-clean-all', dest='clean_all', action='store_false',
                        help='Tắt tùy chọn làm sạch tất cả')
    
    # Tùy chọn làm sạch tự động
    parser.add_argument('--auto-clean', action='store_true', default=False,
                        help='Sử dụng tính năng tự động làm sạch dựa trên phân tích dữ liệu')
    
    # Các tùy chọn xử lý dữ liệu
    parser.add_argument('--no-duplicates', action='store_true',
                        help='Loại bỏ các bản ghi trùng lặp')
    
    parser.add_argument('--handle-outliers', choices=['none', 'remove', 'cap', 'iqr', 'zscore'],
                        default='none', help='Phương pháp xử lý outliers')
    
    parser.add_argument('--handle-missing', choices=['remove', 'fill_zero', 'fill_mean', 'fill_median', 'fill_mode', 'ffill', 'bfill'],
                        default='remove', help='Phương pháp xử lý giá trị thiếu')
    
    parser.add_argument('--normalize-text', action='store_true',
                        help='Chuẩn hóa các cột văn bản')

    parser.add_argument('--normalize-phone', action='store_true',
                        help='Chuẩn hóa các cột số điện thoại')

    parser.add_argument('--normalize-address', action='store_true',
                        help='Chuẩn hóa các cột địa chỉ')
    
    parser.add_argument('--case-consistency', choices=['none', 'lower', 'upper', 'title'],
                        default='none', help='Phương pháp xử lý tính nhất quán chữ hoa/thường')
    
    # Tùy chọn cho việc xử lý theo lô
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Kích thước của lô dữ liệu khi xử lý (mặc định: 5000 dòng)')
    
    parser.add_argument('--use-batches', action='store_true', default=True,
                        help='Sử dụng xử lý theo lô (mặc định: bật)')
    
    # Các tùy chọn cho báo cáo
    parser.add_argument('--generate-report', action='store_true', default=True,
                        help='Tạo báo cáo HTML tóm tắt quá trình làm sạch (mặc định bật)')
    
    parser.add_argument('--no-report', dest='generate_report', action='store_false',
                        help='Không tạo báo cáo HTML')
    
    parser.add_argument('--report-path', type=str, default='data/reports',
                        help='Đường dẫn đến thư mục lưu báo cáo')
    
    parser.add_argument('--simple-report', action='store_true', default=True,
                        help='Sử dụng báo cáo đơn giản với ít định dạng hơn')
    
    # Tùy chọn về logging
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Mức độ chi tiết của log')
    
    # Tùy chọn định dạng dữ liệu - mới
    parser.add_argument('--optimize-datatypes', action='store_true', default=False,
                       help='Tối ưu hóa kiểu dữ liệu để giảm dung lượng bộ nhớ')
                       
    parser.add_argument('--auto-format', action='store_true', default=True,
                       help='Tự động áp dụng định dạng phù hợp cho từng kiểu dữ liệu (mặc định: bật)')
                       
    parser.add_argument('--no-auto-format', dest='auto_format', action='store_false',
                       help='Không tự động áp dụng định dạng')
                       
    parser.add_argument('--save-metadata', action='store_true', default=True,
                      help='Lưu metadata về kiểu dữ liệu cùng với file CSV (mặc định: bật)')
                      
    parser.add_argument('--no-save-metadata', dest='save_metadata', action='store_false',
                      help='Không lưu metadata về kiểu dữ liệu')
    
    args = parser.parse_args()
    
    # Nếu --clean-all được bật, kích hoạt tất cả các tùy chọn làm sạch
    if args.clean_all and not args.auto_clean:
        args.no_duplicates = True
        args.handle_outliers = 'iqr'  # Phương pháp tốt cho hầu hết các trường hợp
        args.handle_missing = 'fill_median'  # Phương pháp an toàn cho cả số và chuỗi
        args.normalize_text = True
        args.normalize_phone = True
        args.normalize_address = True
        args.case_consistency = 'title'  # Chuẩn hóa viết hoa các chữ cái đầu từ
    
    return args

def ensure_output_directory(output_path: str) -> Tuple[bool, str]:
    """
    Đảm bảo thư mục đầu ra tồn tại
    
    Parameters:
    -----------
    output_path : str
        Đường dẫn đến file đầu ra
    
    Returns:
    --------
    Tuple[bool, str]
        (True nếu thư mục tồn tại hoặc được tạo thành công, False nếu có lỗi, đường dẫn đầu ra mới hoặc gốc)
    """
    # Lấy đường dẫn tuyệt đối và thư mục cha
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    
    # Nếu đường dẫn không có thư mục cha (file trong thư mục hiện tại)
    if not output_dir:
        print(f"File sẽ được lưu trong thư mục hiện tại: {os.getcwd()}")
        return True, output_path
    
    try:
        # Tạo thư mục đệ quy (tương tự mkdir -p)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Đã tạo thư mục đầu ra: {output_dir}")
        logging.info(f"Đã tạo thư mục đầu ra: {output_dir}")
        
        # Kiểm tra quyền ghi vào thư mục
        if os.access(output_dir, os.W_OK):
            return True, output_path
        else:
            print(f"Cảnh báo: Có thể không có quyền ghi vào thư mục {output_dir}")
            logging.warning(f"Có thể không có quyền ghi vào thư mục {output_dir}")
            return True, output_path  # Vẫn trả về True, để thử lưu
            
    except PermissionError:
        logging.error(f"Lỗi quyền truy cập: Không thể tạo thư mục {output_dir}")
        print(f"Lỗi: Không có quyền tạo thư mục {output_dir}")
        
        # Thử tạo trong thư mục hiện tại
        try:
            alt_dir = os.path.join(os.getcwd(), "output")
            Path(alt_dir).mkdir(exist_ok=True)
            logging.warning(f"Chuyển hướng đầu ra về thư mục: {alt_dir}")
            print(f"Đã tạo thư mục thay thế: {alt_dir}")
            
            # Cập nhật lại đường dẫn trong các tham số
            new_output_path = os.path.join(alt_dir, os.path.basename(output_path))
            print(f"File sẽ được lưu tại: {new_output_path}")
            logging.info(f"File sẽ được lưu tại: {new_output_path}")
            
            return True, new_output_path
        except Exception as e2:
            logging.error(f"Không thể tạo thư mục thay thế: {str(e2)}")
            print(f"Lỗi: Không thể tạo thư mục thay thế: {str(e2)}")
            return False, output_path
    except Exception as e:
        logging.error(f"Lỗi khi tạo thư mục đầu ra {output_dir}: {str(e)}")
        print(f"Lỗi: Không thể tạo thư mục đầu ra {output_dir}: {str(e)}")
        return False, output_path

def generate_sample_data(rows=100):
    """Tạo dữ liệu mẫu để thử nghiệm"""
    np.random.seed(42)
    
    # Tạo DataFrame mẫu
    data = {
        'Mã SP': [f'SP{i:04d}' for i in range(1, rows+1)],
        'Tên sản phẩm': [f'Sản phẩm thử nghiệm {i}' for i in range(1, rows+1)],
        'Giá': np.random.randint(50000, 2000000, size=rows),
        'Số lượng': np.random.randint(0, 100, size=rows),
        'Ngày nhập': pd.date_range(start='2023-01-01', periods=rows),
        'Đánh giá': np.random.randint(1, 6, size=rows),
        'Mô tả': [f'Đây là mô tả cho sản phẩm thử nghiệm {i}' for i in range(1, rows+1)]
    }
    
    # Thêm một số giá trị thiếu
    df = pd.DataFrame(data)
    mask = np.random.random(size=df.shape) < 0.1  # 10% giá trị thiếu
    df = df.mask(mask)
    
    return df

def main() -> int:
    """
    Hàm chính điều khiển quá trình làm sạch dữ liệu
    
    Returns:
    --------
    int
        Mã trạng thái: 0 nếu thành công, khác 0 nếu có lỗi
    """
    try:
        # Phân tích tham số dòng lệnh
        args = parse_arguments()
        
        # Hiển thị thông tin về chế độ làm sạch
        if args.auto_clean:
            mode_message = "CHẾ ĐỘ LÀM SẠCH TỰ ĐỘNG"
        else:
            mode_message = "CHẾ ĐỘ LÀM SẠCH TOÀN DIỆN" if args.clean_all else "CHẾ ĐỘ LÀM SẠCH TÙY CHỈNH"
        
        # Thiết lập logging với mức độ tùy chỉnh
        log_file = setup_logging(log_level=args.log_level.upper())
        
        # Hiển thị thông tin đầu vào
        print("\n" + "="*50)
        print("PHẦN MỀM LÀM SẠCH DỮ LIỆU CSV")
        print("="*50)
        print(f"=> {mode_message}")
        print(f"- File đầu vào: {args.input}")
        print(f"- File đầu ra: {args.output}")
        print(f"- Mã hóa: {args.encoding}")
        if not args.auto_clean:
            print(f"- Phương pháp xử lý missing: {args.handle_missing}")
            print(f"- Phương pháp xử lý outliers: {args.handle_outliers}")
        
        if args.auto_clean:
            print("- Tự động làm sạch dựa trên phân tích dữ liệu đã được kích hoạt")
        elif args.clean_all:
            print("- Tất cả các tùy chọn làm sạch đã được kích hoạt")
        
        print("="*50 + "\n")
        
        # Đảm bảo thư mục đầu ra tồn tại
        save_success, output_path = ensure_output_directory(args.output)
        
        if not save_success:
            print("Không thể tiếp tục do lỗi thư mục đầu ra.")
            return 1
            
        # Chuẩn bị dict để lưu thông tin cho báo cáo
        report_data = {
            'input_file': args.input,
            'output_file': output_path,
            'log_file': log_file,
            'processing_steps': [],
            'stats': {
                'before': {},
                'after': {}
            }
        }
        
        # Đọc dữ liệu đầu vào
        try:
            print("1. Đang đọc dữ liệu đầu vào...")
            logging.info(f"Đọc dữ liệu từ: {args.input}")
            df, load_info = load_data(
                args.input, 
                sep=args.sep, 
                encoding=args.encoding, 
                na_values=args.na_values
            )
            
            if df is None:
                print(f"Lỗi: Không thể đọc file {args.input}")
                logging.error(f"Không thể đọc file {args.input}")
                return 1
                
            print(f"   - Đã đọc thành công dữ liệu với {df.shape[0]:,} dòng và {df.shape[1]} cột.")
            logging.info(f"Đã đọc thành công: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Kiểm tra dữ liệu đã đọc
            if df.empty:
                print("Cảnh báo: DataFrame trống, không có gì để làm sạch.")
                logging.warning("DataFrame trống, không có gì để làm sạch")
                return 0
                
            # Phát hiện vấn đề dữ liệu
            print("\n1.1 Đang phát hiện vấn đề dữ liệu...")
            data_issues = detect_data_issues(df)
            report_data['data_issues'] = data_issues
                
        except Exception as e:
            print(f"Lỗi khi đọc dữ liệu: {str(e)}")
            logging.exception("Lỗi khi đọc dữ liệu")
            return 1
        
        # Lưu bản sao dữ liệu gốc cho báo cáo
        df_original = df.copy()
        
        # Lưu thông tin dữ liệu ban đầu để so sánh
        original_shape = df.shape
        original_missing = df.isnull().sum().sum()
        report_data['stats']['before'] = get_summary_stats(df)
        
        # Xử lý làm sạch dữ liệu
        if args.auto_clean:
            print("\n2. Đang phân tích dữ liệu và tạo khuyến nghị làm sạch...")
            analysis_result = analyze_and_suggest(df)
            print("\n3. Đang tự động làm sạch dữ liệu theo khuyến nghị...")
            df, cleaning_report = auto_clean_data(df, analysis_result)
            report_data['processing_steps'] = cleaning_report['actions_taken']
            report_data['analysis_result'] = analysis_result
        else:
            # Xử lý dữ liệu thiếu
            print("\n2. Đang xử lý giá trị thiếu...")
            logging.info(f"Bắt đầu xử lý giá trị thiếu với phương pháp: {args.handle_missing}")
            
            if args.handle_missing == 'remove':
                df = delete_missing_values(df, how='any')
                report_data['processing_steps'].append({
                    'step': 'Xử lý giá trị thiếu',
                    'method': 'remove',
                    'details': f"Đã xóa các dòng có giá trị thiếu"
                })
            elif args.handle_missing.startswith('fill_'):
                strategy = args.handle_missing.split('_')[1]  # Lấy phần sau fill_
                df = fill_missing_values(df, strategy=strategy)
                report_data['processing_steps'].append({
                    'step': 'Xử lý giá trị thiếu',
                    'method': f'fill_{strategy}',
                    'details': f"Đã điền giá trị thiếu bằng phương pháp {strategy}"
                })
            elif args.handle_missing in ['ffill', 'bfill']:
                df = fill_missing_values(df, strategy=args.handle_missing)
                report_data['processing_steps'].append({
                    'step': 'Xử lý giá trị thiếu',
                    'method': args.handle_missing,
                    'details': f"Đã điền giá trị thiếu bằng phương pháp {args.handle_missing}"
                })
                
            logging.info(f"Sau khi xử lý giá trị thiếu: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Chuẩn hóa văn bản
            if args.normalize_text:
                print("\n3. Đang chuẩn hóa văn bản...")
                logging.info("Bắt đầu chuẩn hóa văn bản")
                df = normalize_text(df, batch_size=args.batch_size if args.use_batches else 0)
                report_data['processing_steps'].append({
                    'step': 'Chuẩn hóa văn bản',
                    'method': 'normalize_text',
                    'details': "Đã chuẩn hóa các cột văn bản (loại bỏ khoảng trắng thừa, dấu câu thừa)"
                })
                logging.info(f"Sau khi chuẩn hóa văn bản: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Tính nhất quán chữ hoa/thường
            if args.case_consistency != 'none':
                print(f"\n3.1 Đang xử lý tính nhất quán chữ hoa/thường với phương pháp {args.case_consistency}...")
                df = handle_case_consistency(df, case=args.case_consistency)
                report_data['processing_steps'].append({
                    'step': 'Xử lý tính nhất quán chữ hoa/thường',
                    'method': args.case_consistency,
                    'details': f"Đã chuyển các chuỗi văn bản sang dạng {args.case_consistency}"
                })
                
            # Chuẩn hóa số điện thoại nếu được yêu cầu
            if args.normalize_phone:
                print("\n3.2 Đang chuẩn hóa số điện thoại...")
                # Tự động phát hiện các cột số điện thoại dựa trên tên
                phone_columns = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['phone', 'tel', 'mobile', 'số điện thoại', 'di động', 'sdt']):
                        phone_columns.append(col)
                
                if phone_columns:
                    print(f"   - Đã tìm thấy {len(phone_columns)} cột số điện thoại: {phone_columns}")
                    for phone_col in phone_columns:
                        df = standardize_phone_numbers(df, column=phone_col)
                    report_data['processing_steps'].append({
                        'step': 'Chuẩn hóa số điện thoại',
                        'method': 'standardize_phone_numbers',
                        'columns': phone_columns,
                        'details': f"Đã chuẩn hóa {len(phone_columns)} cột số điện thoại"
                    })
                else:
                    print("   - Không tìm thấy cột số điện thoại nào dựa vào tên cột")
                
            # Chuẩn hóa địa chỉ nếu được yêu cầu
            if args.normalize_address:
                print("\n3.3 Đang chuẩn hóa địa chỉ...")
                # Tự động phát hiện các cột địa chỉ dựa trên tên
                address_columns = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['address', 'địa chỉ', 'location', 'vị trí']):
                        address_columns.append(col)
                
                if address_columns:
                    print(f"   - Đã tìm thấy {len(address_columns)} cột địa chỉ: {address_columns}")
                    for addr_col in address_columns:
                        df = standardize_addresses(df, column=addr_col)
                    report_data['processing_steps'].append({
                        'step': 'Chuẩn hóa địa chỉ',
                        'method': 'standardize_addresses',
                        'columns': address_columns,
                        'details': f"Đã chuẩn hóa {len(address_columns)} cột địa chỉ"
                    })
                else:
                    print("   - Không tìm thấy cột địa chỉ nào dựa vào tên cột")
            
            # Chuẩn hóa ngày tháng - Tìm các cột ngày tháng
            print("\n4. Đang chuẩn hóa ngày tháng...")
            logging.info("Bắt đầu chuẩn hóa ngày tháng")
            
            # Tìm các cột có thể là ngày tháng dựa vào tên
            date_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'ngay', 'thang', 'nam']):
                    date_columns.append(col)
            
            # Chuẩn hóa từng cột ngày tháng
            if date_columns:
                print(f"   - Đã tìm thấy {len(date_columns)} cột ngày tháng: {date_columns}")
                for date_col in date_columns:
                    df = standardize_dates(df, column=date_col, errors='coerce')
                report_data['processing_steps'].append({
                    'step': 'Chuẩn hóa ngày tháng',
                    'method': 'standardize_dates',
                    'columns': date_columns,
                    'details': f"Đã chuẩn hóa {len(date_columns)} cột ngày tháng"
                })
            else:
                print("   - Không tìm thấy cột ngày tháng nào dựa vào tên cột")
            
            logging.info(f"Sau khi chuẩn hóa ngày tháng: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Chuyển đổi kiểu dữ liệu
            print("\n5. Đang chuyển đổi kiểu dữ liệu...")
            logging.info("Bắt đầu chuyển đổi kiểu dữ liệu")
            
            # Tạo ánh xạ kiểu dữ liệu dựa trên phân tích dữ liệu
            numeric_cols = []
            categorical_cols = []
            
            for col in df.columns:
                # Bỏ qua các cột đã là datetime
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                    
                # Kiểm tra xem cột có thể chuyển đổi thành số không
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                elif df[col].nunique() < len(df) * 0.1:  # Ít hơn 10% giá trị duy nhất
                    categorical_cols.append(col)
            
            # Chuyển đổi các cột kiểu số
            if numeric_cols:
                print(f"   - Chuyển đổi {len(numeric_cols)} cột sang kiểu số")
                df = convert_to_numeric(df, columns=numeric_cols, errors='coerce')
                report_data['processing_steps'].append({
                    'step': 'Chuyển đổi kiểu dữ liệu',
                    'method': 'convert_to_numeric',
                    'columns': numeric_cols,
                    'details': f"Đã chuyển đổi {len(numeric_cols)} cột sang kiểu số"
                })
                
            # Chuyển đổi các cột phân loại
            if categorical_cols:
                print(f"   - Chuyển đổi {len(categorical_cols)} cột sang kiểu category")
                df = convert_to_category(df, columns=categorical_cols)
                report_data['processing_steps'].append({
                    'step': 'Chuyển đổi kiểu dữ liệu',
                    'method': 'convert_to_category',
                    'columns': categorical_cols,
                    'details': f"Đã chuyển đổi {len(categorical_cols)} cột sang kiểu category"
                })
            
            logging.info(f"Sau khi chuyển đổi kiểu dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Xử lý bản ghi trùng lặp nếu yêu cầu
            if args.no_duplicates:
                print("\n6. Đang xử lý bản ghi trùng lặp...")
                logging.info("Bắt đầu xử lý bản ghi trùng lặp")
                original_rows = df.shape[0]
                df = remove_duplicates(df)
                removed_rows = original_rows - df.shape[0]
                print(f"   - Đã loại bỏ {removed_rows} bản ghi trùng lặp.")
                report_data['processing_steps'].append({
                    'step': 'Xử lý bản ghi trùng lặp',
                    'method': 'remove_duplicates',
                    'details': f"Đã loại bỏ {removed_rows} bản ghi trùng lặp"
                })
                logging.info(f"Đã loại bỏ {removed_rows} bản ghi trùng lặp")
                logging.info(f"Sau khi xử lý trùng lặp: {df.shape[0]} dòng x {df.shape[1]} cột")
            
            # Xử lý outliers
            if args.handle_outliers != 'none':
                print("\n7. Đang xử lý giá trị ngoại lai...")
                logging.info(f"Bắt đầu xử lý giá trị ngoại lai với phương pháp: {args.handle_outliers}")
                
                # Áp dụng xử lý outlier cho các cột số
                numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
                
                if numeric_columns:
                    for col in numeric_columns:
                        # Bỏ qua các cột không thích hợp cho xử lý outlier (ví dụ: ID, mã, năm...)
                        if any(keyword in col.lower() for keyword in ['id', 'code', 'year', 'nam', 'mã']):
                            continue
                            
                        print(f"   - Đang xử lý outliers trong cột {col}...")
                        original_values = df[col].notna().sum()
                        df = handle_outliers(df, column=col, method=args.handle_outliers)
                        remaining_values = df[col].notna().sum()
                        if original_values != remaining_values:
                            print(f"     + Đã xử lý {original_values - remaining_values} giá trị ngoại lai")
                    
                    report_data['processing_steps'].append({
                        'step': 'Xử lý giá trị ngoại lai',
                        'method': args.handle_outliers,
                        'columns': numeric_columns,
                        'details': f"Đã xử lý giá trị ngoại lai trên các cột số với phương pháp {args.handle_outliers}"
                    })
                else:
                    print("   - Không tìm thấy cột số nào để xử lý outliers")
                    
                logging.info(f"Sau khi xử lý outliers: {df.shape[0]} dòng x {df.shape[1]} cột")
        
        # Lưu thông tin dữ liệu sau khi xử lý để so sánh
        cleaned_shape = df.shape
        cleaned_missing = df.isnull().sum().sum()
        report_data['stats']['after'] = get_summary_stats(df)
        
        # Thêm: Bổ sung các bước tối ưu và định dạng
        if args.optimize_datatypes:
            print("\n8. Đang tối ưu hóa kiểu dữ liệu...")
            logging.info("Bắt đầu tối ưu hóa kiểu dữ liệu")
            memory_before = df.memory_usage(deep=True).sum() / (1024 * 1024)
            df = optimize_datatypes(df)
            memory_after = df.memory_usage(deep=True).sum() / (1024 * 1024)
            report_data['processing_steps'].append({
                'step': 'Tối ưu hóa kiểu dữ liệu',
                'details': f"Đã tối ưu hóa kiểu dữ liệu, giảm {memory_before - memory_after:.2f} MB"
            })
            logging.info(f"Đã tối ưu hóa kiểu dữ liệu: {memory_before:.2f} MB -> {memory_after:.2f} MB")
        
        # Áp dụng định dạng dữ liệu tự động
        if args.auto_format:
            print("\n8.1 Đang áp dụng định dạng tự động...")
            logging.info("Bắt đầu áp dụng định dạng tự động")
            
            # Kiểm tra định dạng hiện tại
            format_issues = validate_data_formats(df)
            
            # Nếu có vấn đề với định dạng, áp dụng các gợi ý
            inconsistent_count = sum(1 for col_result in format_issues.values() if not col_result['format_consistent'])
            
            if inconsistent_count > 0:
                print(f"   - Phát hiện {inconsistent_count} cột có vấn đề định dạng")
                format_suggestions = suggest_column_formats(df)
                df = apply_suggested_formats(df, format_suggestions)
                report_data['processing_steps'].append({
                    'step': 'Chuẩn hóa định dạng dữ liệu',
                    'details': f"Đã áp dụng định dạng phù hợp cho {inconsistent_count} cột"
                })
                logging.info(f"Đã áp dụng định dạng cho {inconsistent_count} cột")
            else:
                print("   - Không phát hiện vấn đề định dạng, các cột đã nhất quán")
                # Vẫn lưu metadata định dạng cho các cột
                df = apply_suggested_formats(df)
                report_data['processing_steps'].append({
                    'step': 'Lưu metadata định dạng',
                    'details': "Đã lưu metadata định dạng cho các cột"
                })
                
        # Xuất thông tin về quá trình làm sạch
        print("\n" + "="*50)
        print("KẾT QUẢ LÀM SẠCH DỮ LIỆU")
        print("="*50)
        print(f"- Dữ liệu ban đầu: {original_shape[0]:,} dòng x {original_shape[1]} cột")
        print(f"- Dữ liệu sau khi làm sạch: {cleaned_shape[0]:,} dòng x {cleaned_shape[1]} cột")
        print(f"- Dữ liệu đã thay đổi: {original_shape[0] - cleaned_shape[0]:,} dòng đã bị xóa")
        print(f"- Dữ liệu thiếu: {original_missing:,} -> {cleaned_missing:,} ô")
        
        if args.auto_clean:
            print("\nTóm tắt quy trình làm sạch tự động:")
            for action in cleaning_report['actions_taken']:
                print(f"- {action['step']}: {action['details']}")
        
        print("="*50)
        
        # Lưu DataFrame đã làm sạch
        print("\n9. Đang lưu dữ liệu đã làm sạch...")
        logging.info(f"Đang lưu dữ liệu đã làm sạch vào: {output_path}")
        
        # Kiểm tra lại thư mục đầu ra trước khi lưu
        save_success, output_path = ensure_output_directory(output_path)
        
        if not save_success:
            print("Lỗi: Không thể lưu dữ liệu đã làm sạch!")
            logging.error("Không thể lưu dữ liệu đã làm sạch")
            return 1
        
        # Lưu dữ liệu với đường dẫn đã được kiểm tra
        save_success = save_data(
            df, output_path, 
            index=False, 
            encoding=args.encoding, 
            sep=args.sep,
            excel=args.excel,
            sheet_name=args.sheet_name,
            save_metadata=args.save_metadata  # Thêm tham số này
        )
        
        if not save_success:
            print("Lỗi: Không thể lưu dữ liệu đã làm sạch!")
            logging.error("Không thể lưu dữ liệu đã làm sạch")
            return 1
        
        # Hiển thị thông tin định dạng đầu ra
        output_format = "Excel (.xlsx)" if args.excel else f"CSV (separator: '{args.sep}')"
        print(f"Định dạng đầu ra: {output_format}")
        if not args.excel and args.save_metadata:
            metadata_path = f"{os.path.splitext(output_path)[0]}.meta.json"
            print(f"Metadata định dạng: {metadata_path}")
        
        # Tạo báo cáo nếu được yêu cầu
        if args.generate_report:
            print("\n10. Đang tạo báo cáo HTML...")
            try:
                # Đảm bảo thư mục báo cáo tồn tại
                report_dir_exists, _ = ensure_output_directory(os.path.join(args.report_path, "report.html"))
                
                if report_dir_exists:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"data_cleaning_report_{timestamp}.html"
                    report_path = os.path.join(args.report_path, report_filename)
                    
                    # Tạo báo cáo HTML
                    report_success = simplify_html_report(
                        report_data, 
                        output_path=report_path
                    )
                    
                    if report_success:
                        print(f"   - Báo cáo đã được tạo tại: {report_path}")
                        logging.info(f"Báo cáo đã được tạo tại: {report_path}")
                        
                        # Xuất thêm báo cáo dạng JSON
                        json_path = os.path.join(args.report_path, f"data_cleaning_report_{timestamp}.json")
                        json_success = export_json_report(report_data, output_path=json_path)
                        
                        if json_success:
                            print(f"   - Báo cáo JSON đã được tạo tại: {json_path}")
                            logging.info(f"Báo cáo JSON đã được tạo tại: {json_path}")
                    else:
                        print("   - Không thể tạo báo cáo HTML")
                        logging.error("Không thể tạo báo cáo HTML")
            except Exception as e:
                print(f"Lỗi khi tạo báo cáo: {str(e)}")
                logging.exception("Lỗi khi tạo báo cáo")
        
        # Hiển thị tóm tắt
        print("\n" + "="*50)
        print("TÓM TẮT QUÁ TRÌNH LÀM SẠCH DỮ LIỆU")
        print("="*50)
        print(f"- Số dòng ban đầu: {original_shape[0]:,}")
        print(f"- Số dòng sau khi làm sạch: {cleaned_shape[0]:,}")
        print(f"- Số cột: {cleaned_shape[1]}")
        print(f"- Số giá trị thiếu ban đầu: {original_missing:,}")
        print(f"- Số giá trị thiếu sau khi làm sạch: {cleaned_missing:,}")
        print(f"- File đầu ra: {output_path}")
        print(f"- Định dạng đầu ra: {output_format}")
        
        if args.generate_report:
            print(f"- Báo cáo: {report_path}")
            
        print("\nĐã hoàn thành quá trình làm sạch dữ liệu!")
        print("="*50)
        
        logging.info("Hoàn thành quá trình làm sạch dữ liệu")
        logging.info(f"Tóm tắt: {original_shape[0]} dòng ban đầu -> {cleaned_shape[0]} dòng sau khi làm sạch")
        logging.info(f"Giá trị thiếu: {original_missing} ban đầu -> {cleaned_missing} sau khi làm sạch")
        logging.info(f"Định dạng đầu ra: {output_format}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nQuá trình bị ngắt bởi người dùng.")
        logging.warning("Quá trình bị ngắt bởi người dùng")
        return 130
    except Exception as e:
        print(f"\nLỗi không xác định: {str(e)}")
        logging.exception("Lỗi không xác định")
        return 1

if __name__ == "__main__":
    sys.exit(main())
