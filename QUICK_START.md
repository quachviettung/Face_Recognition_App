# 🚀 Hướng dẫn khởi động nhanh

## ✅ Ứng dụng đã sẵn sàng!

Ứng dụng nhận diện khuôn mặt với giao diện web đẹp mắt đã được khởi động thành công!

### 🌐 Truy cập ứng dụng
Mở trình duyệt web và truy cập: **http://localhost:5000**

### 📱 Các tính năng chính

#### 1. **Trang chủ** (`/`)
- Xem thống kê tổng quan
- Nhận diện khuôn mặt từ ảnh
- Quản lý danh sách khuôn mặt đã đăng ký

#### 2. **Đăng ký khuôn mặt** (`/register`)
- Thêm khuôn mặt mới vào hệ thống
- Upload ảnh với kéo thả
- Validation tự động

#### 3. **Camera real-time** (`/camera`)
- Khởi động camera
- Phát hiện khuôn mặt liên tục
- Cài đặt chất lượng và độ phân giải

### 🎯 Cách sử dụng

#### **Bước 1: Đăng ký khuôn mặt**
1. Click "Đăng ký khuôn mặt mới" trên trang chủ
2. Nhập tên người
3. Upload ảnh có khuôn mặt rõ ràng
4. Click "Đăng ký khuôn mặt"

#### **Bước 2: Nhận diện khuôn mặt**
1. Trên trang chủ, kéo thả ảnh cần nhận diện
2. Click "Nhận diện"
3. Xem kết quả chi tiết

#### **Bước 3: Sử dụng camera**
1. Click "Mở camera" trên trang chủ
2. Click "Bật camera" để khởi động
3. Điều chỉnh vị trí để camera nhìn rõ khuôn mặt

### 🔧 Cài đặt nâng cao

#### **Thay đổi port**
```python
# Trong file simple_web_app.py
app.run(debug=True, host='0.0.0.0', port=8080)  # Thay đổi port
```

#### **Thay đổi thư mục lưu trữ**
```python
UPLOAD_FOLDER = 'my_uploads'        # Thư mục upload
KNOWN_FACES_FOLDER = 'my_faces'     # Thư mục lưu khuôn mặt
```

### 📁 Cấu trúc thư mục (Đã dọn dẹp)
```
face_recognition_app/
├── simple_web_app.py      # Ứng dụng chính (đang chạy)
├── requirements.txt        # Dependencies
├── templates/             # Giao diện HTML
├── uploads/              # Ảnh tạm thời
├── known_faces/          # Ảnh đã đăng ký
├── README.md             # Hướng dẫn chi tiết
└── QUICK_START.md        # Hướng dẫn nhanh (này)
```

### 🚨 Xử lý lỗi thường gặp

#### **Ứng dụng không khởi động**
```bash
# Kiểm tra port đang sử dụng
netstat -an | findstr :5000

# Khởi động lại ứng dụng
python simple_web_app.py
```

#### **Không thể upload ảnh**
- Kiểm tra định dạng file (JPG, PNG, GIF)
- Đảm bảo ảnh có khuôn mặt rõ ràng
- Kiểm tra quyền ghi thư mục

#### **Camera không hoạt động**
- Cho phép trình duyệt truy cập camera
- Kiểm tra camera có được kết nối
- Thử trình duyệt khác

### 🔮 Nâng cấp lên phiên bản đầy đủ

Để có tính năng nhận diện khuôn mặt chính xác hơn:

1. **Cài đặt Visual Studio Build Tools**
2. **Cài đặt CMake từ cmake.org**
3. **Cài đặt dlib và face_recognition**
4. **Tích hợp face_recognition vào simple_web_app.py**

### 📞 Hỗ trợ

- **Ứng dụng đang chạy**: http://localhost:5000
- **Trạng thái**: ✅ Hoạt động
- **Port**: 5000
- **Giao diện**: Responsive, hiện đại
- **Cấu trúc**: ✅ Đã dọn dẹp, gọn gàng

---

## 🎉 Chúc mừng! Bạn đã có một hệ thống nhận diện khuôn mặt hoàn chỉnh và gọn gàng!

Hãy truy cập **http://localhost:5000** để bắt đầu sử dụng ngay!
