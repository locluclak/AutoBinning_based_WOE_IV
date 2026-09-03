# Công cụ chia bin tự động cho biến đặc trưng liên tục

_*English below*_

**Đọc tài liệu [Công cụ chia bin tự động cho biến đặc trưng liên tục](https://docs.google.com/document/d/1lYiFhkelPDIuyY7MHQYKfbOA1iIJUUAkinRBY5w71Ro/edit?tab=t.0) để biết chi tiết hơn.**

Trong bài toán phân loại nhị phân, Weight of Evidence (WOE) và Information Value (IV) là hai chỉ số quan trọng được sử dụng để đánh giá và diễn giải mức độ ảnh hưởng của biến đặc trưng đối với biến đầu ra. Đối với biến đặc trưng liên tục, dữ liệu thường cần được chia thành nhiều khoảng, hay bin, để có thể tính WOE và IV theo từng khoảng, đồng thời quan sát rõ hơn xu hướng biến đổi của WOE theo giá trị của biến, chẳng hạn xu hướng đơn điệu (monotonic).

Việc xác định khoảng bin thủ công đòi hỏi người dùng phải trực tiếp phân tích dữ liệu và thử nghiệm nhiều cách chia khác nhau, do đó có thể tốn nhiều thời gian. Công cụ được giới thiệu trong báo cáo này **hỗ trợ đề xuất các khoảng bin phù hợp ngay từ đầu** dựa trên dữ liệu, thay vì yêu cầu người dùng xác định khoảng bin trực tiếp từ dữ liệu thô. Kết quả do công cụ cung cấp có thể được sử dụng làm **giá trị tham khảo** để người dùng tiếp tục đánh giá hoặc tinh chỉnh khoảng bin theo nhu cầu.

Công cụ được xây dựng dựa trên thư viện **OptBinning**, nhưng được đơn giản hóa về cách sử dụng và bổ sung hướng dẫn trực quan nhằm hỗ trợ quá trình thực hiện chia bin và đánh giá kết quả.

## Cài đặt
Tải source code từ repository
```
git clone https://github.com/locluclak/AutoBinning_based_WOE_IV.git
```

Cài các thư viện phụ thuộc vào môi trường:
```
pip install -r requirement.txt
```

## Chia bins theo lô (số lượng lớn)
Cấu hình trong file `config.yaml` bao gồm tên file data, tên biến mục tiêu, các biến cần bỏ qua, các ràng buộc, tên file output.

Thực thi
```
python woe_all_features.py
```

Output sẽ là file HTML theo tên trong config. 

File HTML giúp người dùng chọn và export ra file json thông tin splits bins,.

## Điều chỉnh từng bin thủ công

Dùng file [autobinning_ver3.ipynb](autobinning_ver3.ipynb) để quan sát và điều chỉnh.

## Tái tạo kết quả từ file .json

Chạy dòng lệnh để tái tạo kết quả trong file .json thành HTML.
```
python reconstruct_report.py selected_feature_splits.json output.html
```

## Tạo dữ liệu với transform feature mới
```
python transform_data.py selected_feature_splits.json transformed.csv
```

---


# Automated Binning Tool for Continuous Features

**Read the documentation [Automated Binning Tool for Continuous Features](https://docs.google.com/document/d/1lYiFhkelPDIuyY7MHQYKfbOA1iIJUUAkinRBY5w71Ro/edit?tab=t.0) for more details.**

In binary classification problems, Weight of Evidence (WOE) and Information Value (IV) are two important metrics used to assess and interpret the degree of influence of a feature on the target variable. For continuous features, the data usually needs to be divided into multiple intervals, or bins, so that WOE and IV can be computed per interval, and the trend of WOE across the feature's values can be observed more clearly, e.g., a monotonic trend.



Manually determining bin boundaries requires users to directly analyze the data and experiment with many different binning schemes, which can be very time-consuming. The tool introduced in this report **helps propose suitable bin boundaries from the start** based on the data, rather than requiring users to determine bin boundaries directly from raw data. The results provided by the tool can be used as a **reference value** for users to further evaluate or fine-tune the bin boundaries as needed.



The tool is built on top of the **OptBinning** library, but with a simplified usage and added visual guidance to support the binning process and evaluation of results.



## Installation
Download the source code from the repository:
```
git clone https://github.com/locluclak/AutoBinning_based_WOE_IV.git
```

Install the required dependencies into your environment:
```
pip install -r requirement.txt
```

## Binning in batch (large numbers)
Configure `config.yaml` including the data file name, target variable name, features to ignore, constraints, and output file name.



Run:
```
python woe_all_features.py
```



The output will be an HTML file named according to the config.



The HTML file helps users select and export the split/bin information to a JSON file.



## Manually adjusting individual bins

Use the [autobinning_ver3.ipynb](autobinning_ver3.ipynb) notebook to observe and adjust.



## Reproducing results from a .json file

Run the following command to reproduce the results in the .json file as HTML:
```
python reconstruct_report.py selected_feature_splits.json output.html
```



## Generating data with new transformed features
```
python transform_data.py selected_feature_splits.json transformed.csv
```

