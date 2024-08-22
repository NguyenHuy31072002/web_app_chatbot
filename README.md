# CHATBOT LUẬT VIỆT NAM
## Overview
Kho lưu trữ này chứa cách triển khai cơ bản của một ứng dụng web bằng API chatbot Gemini AI. Ứng dụng web được xây dựng bằng khung Flask trong Python. Chatbot Gemini AI được cung cấp bởi API Gemini Pro, cho phép người dùng tương tác với chatbot được đào tạo trên bộ dữ liệu khổng lồ gồm 1,5 nghìn tỷ mã thông báo. 
Xin lưu ý rằng API Gemini cho phép 60 truy vấn mỗi phút.

## Local Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/NguyenHuy31072002/web_app_chatbot.git
```

### Step 2: Add API Key
Replace the following line of code in app.py with your Gemini API key:
```python
my_api_key_gemini = os.getenv('my_new_api_key_here')
```
Replace it with:

```python
my_api_key_gemini = "your_api_key_here"
```
You can obtain your API key from [here](https://makersuite.google.com/app/apikey). Once you have the key, proceed to the next step.


### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Run the Web App
In the terminal, execute the following command:

```
python app.py
```

This will launch a web app locally. Open your browser and navigate to the provided address (usually http://localhost:5000/) to interact with the Gemini AI chatbot.

Feel free to explore and customize the code according to your needs. Contributions are welcome!

**Notes**
* Ensure that you have Python installed on your system.
* Use a virtual environment for better isolation of dependencies.
