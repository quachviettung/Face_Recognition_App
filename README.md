# Face Recognition Web Application

Ứng dụng web nhận diện khuôn mặt sử dụng OpenCV và Python Flask.

## Tính năng

- ✅ Đăng ký khuôn mặt mới
- ✅ Nhận diện khuôn mặt từ ảnh
- ✅ Giao diện web thân thiện
- ✅ Hỗ trợ nhiều định dạng ảnh (JPG, PNG, JPEG, GIF)
- ✅ Lưu trữ dữ liệu bền vững (JSON)
- ✅ Debug logging chi tiết

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- OpenCV
- Flask
- NumPy

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Sử dụng

### Khởi động ứng dụng
```bash
python simple_web_app.py
```

### Truy cập ứng dụng
Mở trình duyệt và truy cập: http://localhost:5000

### Các chức năng chính

1. **Đăng ký khuôn mặt**
   - Truy cập `/register`
   - Nhập tên và upload ảnh
   - Hệ thống sẽ tự động phát hiện khuôn mặt

2. **Nhận diện khuôn mặt**
   - Truy cập trang chủ
   - Upload ảnh cần nhận diện
   - Xem kết quả nhận diện

3. **Quản lý khuôn mặt**
   - Xem danh sách khuôn mặt đã đăng ký
   - Xóa khuôn mặt không cần thiết

## Cấu trúc dự án

```
face_recognition_app/
├── simple_web_app.py      # File chính của ứng dụng
├── requirements.txt        # Dependencies
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   └── camera.html
├── uploads/               # Thư mục lưu ảnh upload
├── known_faces/           # Database khuôn mặt
│   └── known_faces.json
└── README.md
```

## Thuật toán nhận diện

Ứng dụng sử dụng OpenCV với Haar Cascade để phát hiện khuôn mặt và thuật toán so sánh đa phương pháp:

1. **Histogram Comparison** (30%)
2. **Template Matching** (30%)
3. **Structural Similarity Index** (25%)
4. **Edge Detection** (15%)

## Cải tiến gần đây

- ✅ Tăng độ chính xác nhận diện
- ✅ Cải thiện thuật toán so sánh
- ✅ Thêm debug logging chi tiết
- ✅ Tối ưu hóa tham số phát hiện khuôn mặt
- ✅ Tăng ngưỡng nhận diện để tránh sai sót

## Tác giả

**Quách Việt Tùng** - [GitHub](https://github.com/quachviettung)

## License

MIT License

## Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## Hỗ trợ

Nếu gặp vấn đề, hãy tạo issue trên GitHub repository.
