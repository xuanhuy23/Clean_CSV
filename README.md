# Clean-CSV

Công cụ làm sạch dữ liệu CSV mạnh mẽ và dễ sử dụng, được viết bằng Python.

## Giới thiệu

Clean-CSV là một ứng dụng dòng lệnh được thiết kế để tự động hóa quá trình làm sạch và chuẩn hóa dữ liệu từ các file CSV. Ứng dụng cung cấp nhiều tính năng mạnh mẽ như:

- Xử lý giá trị thiếu
- Chuẩn hóa văn bản
- Chuẩn hóa ngày tháng
- Chuyển đổi kiểu dữ liệu
- Phát hiện và loại bỏ bản ghi trùng lặp
- Xử lý giá trị ngoại lai (outliers)
- Ghi log chi tiết cho quá trình xử lý
- Xử lý dữ liệu theo lô (batches) để tối ưu hiệu suất với tập dữ liệu lớn
- Tạo báo cáo HTML và JSON về quá trình làm sạch dữ liệu
- Tự động phát hiện và xử lý vấn đề định dạng ngày tháng

## Cài đặt

1. Clone repository:
```
git clone https://github.com/your-username/Clean-CSV.git
cd Clean-CSV
```

2. Cài đặt các thư viện phụ thuộc:
```
pip install -r requirements.txt
```

## Cấu trúc dự án

```
Clean-CSV/
├── data/                  # Thư mục dữ liệu
│   ├── raw/               # Dữ liệu gốc
│   └── processed/         # Dữ liệu đã xử lý
├── logs/                  # File logs
├── src/                   # Mã nguồn
│   ├── data_loader.py     # Module đọc dữ liệu
│   ├── data_cleaner.py    # Module làm sạch dữ liệu
│   ├── data_validator.py  # Module kiểm tra dữ liệu
│   └── utils.py           # Các hàm tiện ích
├── main.py                # Chương trình chính
├── requirements.txt       # Thư viện phụ thuộc
└── README.md              # Tài liệu dự án
```

## Sử dụng

### Cách đơn giản nhất

```
python main.py --input data/raw/input.csv --output data/processed/output.csv
```

### Tùy chọn nâng cao

```
python main.py --input data/raw/input.csv --output data/processed/output.csv --encoding utf-8 --sep "," --no-duplicates --handle-outliers iqr
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--input` | Đường dẫn đến file CSV đầu vào | `data/raw/input.csv` |
| `--output` | Đường dẫn đến file đầu ra (CSV hoặc Excel) | `data/processed/output.csv` |
| `--encoding` | Mã hóa ký tự của file | `utf-8` |
| `--sep` | Ký tự phân cách cột | `,` |
| `--excel` | Xuất dữ liệu đã làm sạch sang định dạng Excel (.xlsx) | `False` |
| `--sheet-name` | Tên sheet trong file Excel (khi sử dụng --excel) | `Data` |
| `--na-values` | Các giá trị được coi là NA | `['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN']` |
| `--no-duplicates` | Loại bỏ các bản ghi trùng lặp | `False` |
| `--handle-outliers` | Phương pháp xử lý outliers (`none`, `remove`, `cap`, `iqr`, `zscore`) | `none` |
| `--batch-size` | Kích thước lô khi xử lý dữ liệu lớn | `1000` |
| `--use-batches` | Sử dụng xử lý theo lô để tối ưu hiệu suất | `True` |
| `--normalize-text` | Chuẩn hóa các cột văn bản | `False` |
| `--normalize-phone` | Chuẩn hóa số điện thoại | `False` |
| `--normalize-address` | Chuẩn hóa địa chỉ | `False` |
| `--case-consistency` | Phương pháp xử lý chữ hoa/thường (`none`, `lower`, `upper`, `title`) | `none` |

## Quy trình làm sạch dữ liệu

Ứng dụng thực hiện các bước sau khi làm sạch dữ liệu:

1. Đọc dữ liệu đầu vào
2. Xử lý giá trị thiếu
3. Chuẩn hóa văn bản
4. Chuẩn hóa ngày tháng
5. Chuyển đổi kiểu dữ liệu
6. Xử lý bản ghi trùng lặp (nếu được chỉ định)
7. Xử lý giá trị ngoại lai (nếu được chỉ định)
8. Lưu dữ liệu đã làm sạch

## Module chính

### data_loader.py

Module này chịu trách nhiệm đọc dữ liệu từ file CSV và thực hiện các kiểm tra cơ bản.

### data_cleaner.py

Module này chứa các hàm để làm sạch dữ liệu, bao gồm xử lý giá trị thiếu, chuẩn hóa dữ liệu, xử lý giá trị ngoại lai, v.v.

### data_validator.py

Module này chứa các hàm để kiểm tra tính hợp lệ của dữ liệu trước và sau khi làm sạch.

### utils.py

Module này chứa các hàm tiện ích như lưu dữ liệu, tạo thống kê tóm tắt, v.v.

## Ví dụ

Ví dụ về làm sạch tệp CSV chứa dữ liệu bán hàng:

```
python main.py --input data/raw/sales_data.csv --output data/processed/clean_sales_data.csv --no-duplicates --handle-outliers iqr
```

Ví dụ về làm sạch dữ liệu và xuất ra file Excel:

```
python main.py --input data/raw/sales_data.csv --output data/processed/clean_sales_data.xlsx --excel --sheet-name "Sales Data" --normalize-text --handle-outliers iqr
```

## Yêu cầu

- Python 3.6+
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0

## Giấy phép

Dự án này được cấp phép theo [MIT License](LICENSE).

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request trên GitHub.

## Tính năng mới

### Xử lý theo lô (Batch Processing)

Ứng dụng hỗ trợ xử lý dữ liệu theo lô, giúp xử lý hiệu quả các tập dữ liệu lớn mà không gặp vấn đề về bộ nhớ cache:

```bash
python main.py --input large_data.csv --output clean_large_data.csv --batch-size 500
```

Bạn có thể điều chỉnh kích thước lô phù hợp với cấu hình máy tính. Giá trị mặc định là 1000 dòng.

### Xuất file Excel

Giờ đây, bạn có thể xuất dữ liệu đã làm sạch sang định dạng Excel (.xlsx) thay vì CSV:

```bash
python main.py --input data.csv --output data.xlsx --excel --sheet-name "Dữ liệu đã làm sạch"
```

Hoặc đơn giản chỉ cần sử dụng đuôi .xlsx cho file đầu ra:

```bash
python main.py --input data.csv --output data.xlsx
```

### Định dạng dữ liệu thông minh (Mới)

Tính năng mới giúp đảm bảo định dạng dữ liệu được nhất quán và phù hợp với từng kiểu dữ liệu:

- **Tự động nhận diện định dạng**: Hệ thống tự động phát hiện và áp dụng định dạng phù hợp cho từng cột dựa trên kiểu dữ liệu
- **Phân tích và đề xuất định dạng**: Phân tích dữ liệu để đề xuất định dạng phù hợp nhất cho từng cột
- **Định dạng số thông minh**: Tự động phát hiện số chữ số thập phân phù hợp cho từng cột số
- **Định dạng ngày tháng**: Tự động chuyển định dạng ngày tháng về dạng chuẩn (DD/MM/YYYY hoặc DD/MM/YYYY HH:MM:SS)
- **Metadata định dạng**: Lưu thông tin định dạng trong file metadata đi kèm để đảm bảo tính nhất quán khi đọc lại

Sử dụng:

```bash
python main.py --input data.csv --output data.xlsx --auto-format
```

### Tối ưu hóa kiểu dữ liệu

Tiết kiệm bộ nhớ và cải thiện hiệu suất bằng cách tối ưu hóa kiểu dữ liệu:

```bash
python main.py --input data.csv --output data_optimized.csv --optimize-datatypes
```

Tính năng này sẽ:
- Chuyển số nguyên sang kiểu phù hợp nhất (uint8, int16, ...)
- Chuyển float64 sang float32 khi có thể
- Chuyển các cột văn bản có ít giá trị duy nhất sang kiểu category

### Báo cáo đơn giản hóa

Báo cáo HTML đã được đơn giản hóa để tối ưu hiệu suất và dễ đọc. Ngoài ra, ứng dụng còn tạo báo cáo dạng JSON để dễ dàng tích hợp với các hệ thống khác:

```bash
python main.py --input data.csv --output clean_data.csv --generate-report
```

Báo cáo sẽ được lưu trong thư mục `data/reports`.

### Xử lý cải tiến

- **Chuẩn hóa văn bản nâng cao**: Hỗ trợ loại bỏ dấu câu thừa, chuẩn hóa khoảng trắng, và sửa lỗi viết hoa/thường
- **Xử lý outliers linh hoạt**: Thêm phương pháp zscore và các tùy chọn xử lý mới
- **Chuyển đổi kiểu dữ liệu thông minh**: Tự động phát hiện và áp dụng kiểu dữ liệu phù hợp

## Tham số mới

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--auto-format` | Tự động áp dụng định dạng phù hợp cho từng kiểu dữ liệu | `True` |
| `--no-auto-format` | Tắt chức năng tự động định dạng | |
| `--optimize-datatypes` | Tối ưu hóa kiểu dữ liệu để giảm bộ nhớ | `False` |
| `--save-metadata` | Lưu metadata về kiểu dữ liệu cùng với file CSV | `True` |
| `--no-save-metadata` | Không lưu metadata | |
