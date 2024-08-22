# CHATBOT LUẬT VIỆT NAM
## Overview
Kho lưu trữ này chứa cách triển khai cơ bản của một ứng dụng web bằng API chatbot Gemini AI. Ứng dụng web được xây dựng bằng khung Flask trong Python. Chatbot Gemini AI được cung cấp bởi API Gemini Pro, cho phép người dùng tương tác với chatbot được đào tạo trên bộ dữ liệu khổng lồ gồm 1,5 nghìn tỷ mã thông báo. 
Xin lưu ý rằng API Gemini cho phép 60 truy vấn mỗi phút.

## Local Setup

### Bước 1: Sao chép kho lưu trữ
```bash
git clone https://github.com/NguyenHuy31072002/web_app_chatbot.git
```

### Bước 2: Thêm API Key
Replace the following line of code in app.py with your Gemini API key:
```python
my_api_key_gemini = os.getenv('my_new_api_key_here')
```
Replace it with:

```python
my_api_key_gemini = "your_api_key_here"
```
Bạn có thể lấy khóa API của mình từ [tại đây](https://makersuite.google.com/app/apikey). Khi bạn có chìa khóa, hãy chuyển sang bước tiếp theo.


### Bước 3: Cài đặt phụ thuộc
```bash
pip install -r requirements.txt
```
### Bước 4: Chạy Web App
Trong terminal, thực hiện lệnh sau:

```
python app.py
```

Điều này sẽ khởi chạy một ứng dụng web cục bộ. Mở trình duyệt của bạn và điều hướng đến địa chỉ được cung cấp (thường là http://localhost:5000/) để tương tác với chatbot luật pháp việt nam.

Hãy thoải mái khám phá và tùy chỉnh mã theo nhu cầu của bạn. Đóng góp được chào đón!

**Notes**
* Đảm bảo rằng bạn đã cài đặt Python trên hệ thống của mình.
* Sử dụng môi trường ảo để cách ly các phần phụ thuộc tốt hơn.
