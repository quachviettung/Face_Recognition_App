# Face Recognition Web Application

Ứng dụng web nhận diện khuôn mặt sử dụng OpenCV và Python Flask với khả năng nhận diện người thông qua hình ảnh và video.

## Tính năng

### 🎯 Nhận diện khuôn mặt
- ✅ Đăng ký khuôn mặt mới với metadata phong phú
- ✅ Nhận diện khuôn mặt từ ảnh với độ chính xác cao
- ✅ Nhận diện khuôn mặt từ video với xử lý theo thời gian thực
- ✅ Xác minh danh tính khuôn mặt (1:1 comparison)
- ✅ Xử lý batch nhiều ảnh/video cùng lúc
- ✅ Hỗ trợ camera real-time (tương lai)

### 🎨 Giao diện người dùng
- ✅ Giao diện web hiện đại và thân thiện
- ✅ Tab điều hướng cho các chức năng khác nhau
- ✅ Hiển thị kết quả chi tiết với độ tin cậy
- ✅ Upload file bằng drag & drop
- ✅ Xem trước file trước khi xử lý

### 📊 Cơ sở dữ liệu nâng cao
- ✅ Lưu trữ dữ liệu bền vững (JSON)
- ✅ Metadata phong phú (chất lượng ảnh, độ sáng, độ tương phản)
- ✅ Hệ thống tag và ghi chú cho mỗi khuôn mặt
- ✅ Theo dõi lịch sử cập nhật

### ⚡ Hiệu suất và tối ưu hóa
- ✅ Thuật toán nhận diện đa cấp độ tin cậy
- ✅ Tối ưu hóa xử lý ảnh/video
- ✅ Theo dõi hiệu suất và sử dụng bộ nhớ
- ✅ Tự động dọn dẹp file tạm thời
- ✅ Hỗ trợ song song xử lý

### 🔧 Định dạng hỗ trợ
- ✅ **Ảnh**: JPG, PNG, JPEG, GIF
- ✅ **Video**: MP4, AVI, MOV, MKV, WebM

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

#### 1. **Đăng ký khuôn mặt**
   - Truy cập `/register`
   - Nhập tên và upload ảnh
   - Hệ thống sẽ tự động phát hiện và phân tích khuôn mặt
   - Lưu trữ với metadata chất lượng ảnh

#### 2. **Nhận diện khuôn mặt từ ảnh**
   - Chọn tab "Ảnh" trên trang chủ
   - Upload ảnh hoặc kéo thả vào khu vực upload
   - Hệ thống hiển thị kết quả với độ tin cậy

#### 3. **Nhận diện khuôn mặt từ video**
   - Chọn tab "Video" trên trang chủ
   - Upload video (MP4, AVI, MOV, MKV, WebM)
   - Xem kết quả nhận diện theo thời gian và thống kê

#### 4. **Xử lý batch**
   - Chọn tab "Batch" trên trang chủ
   - Upload nhiều ảnh hoặc video cùng lúc
   - Xem báo cáo tổng hợp kết quả xử lý

#### 5. **Xác minh danh tính**
   - Chọn tab "Xác minh" trên trang chủ
   - Chọn người từ danh sách đã đăng ký
   - Upload ảnh cần xác minh
   - Nhận kết quả xác minh với độ tin cậy cao

#### 6. **Camera Real-time**
   - Truy cập `/camera` hoặc click "Mở camera" trên trang chủ
   - Bật camera và cho phép truy cập
   - Hệ thống tự động phát hiện và nhận diện khuôn mặt theo thời gian thực
   - Điều chỉnh các thông số: chất lượng, ngưỡng nhận diện, tần suất xử lý
   - Xem bounding box và thông tin nhận diện trực tiếp trên camera

#### 7. **Quản lý khuôn mặt**
   - Xem danh sách khuôn mặt đã đăng ký với thông tin chi tiết
   - Chỉnh sửa tag và ghi chú cho mỗi khuôn mặt
   - Xóa khuôn mặt không cần thiết
   - Làm mới metadata cho tất cả khuôn mặt

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

## Nguyên lý hoạt động của công nghệ nhận diện khuôn mặt

### 5 Bước thực hiện nhận diện khuôn mặt

#### Bước 1: PHÁT HIỆN (Detection)
- Sử dụng thị giác máy tính để tìm kiếm và xác định vị trí khuôn mặt trong ảnh
- Phát hiện được nhiều khuôn mặt cùng lúc từ các góc độ khác nhau (trực diện, bên cạnh)
- Sử dụng Haar Cascade hoặc Deep Learning models

#### Bước 2: PHÂN TÍCH (Analysis)
- Phân tích chi tiết các đặc điểm khuôn mặt:
  * Khoảng cách giữa hai mắt
  * Khoảng cách từ mũi đến miệng
  * Khoảng cách từ trán đến cằm
  * Hình dạng gò má, độ sâu hốc mắt
  * Đường viền môi, tai, và cằm
- Tạo faceprint (dấu vân tay kỹ thuật số) duy nhất cho mỗi người

#### Bước 3: CHUYỂN ĐỔI DỮ LIỆU (Data Conversion)
- Mã hóa dữ liệu khuôn mặt thành các vector đặc trưng
- Tạo face encodings để lưu trữ và xử lý nhanh chóng
- Chuẩn hóa dữ liệu để so sánh chính xác

#### Bước 4: SO KHỚP DỮ LIỆU (Matching)
- So sánh faceprint với dữ liệu trong cơ sở dữ liệu
- Sử dụng thuật toán học máy và AI để tính độ trùng khớp
- Tính khoảng cách Euclidean giữa các face encodings

#### Bước 5: XÁC NHẬN DANH TÍNH (Verification)
- Xác nhận hoặc từ chối danh tính dựa trên độ trùng khớp
- Trả về kết quả với độ tin cậy (High/Medium/Low)
- Thời gian xử lý: < 2 giây

### Các phương pháp nhận diện khuôn mặt

#### 1. Geometric-Based / Template-Based
- **Phân tích hình học**: Xem xét khoảng cách và mối quan hệ giữa các đặc điểm khuôn mặt
- **Template Matching**: So khớp mẫu khuôn mặt với database
- **Ưu điểm**: Đơn giản, nhanh chóng
- **Nhược điểm**: Nhạy cảm với góc nhìn và ánh sáng

#### 2. Appearance-Based / Model-Based
- **Appearance-Based**: Phân tích hình dáng tổng thể khuôn mặt
- **Model-Based**: Sử dụng mô hình 3D để biểu diễn khuôn mặt
- **Kỹ thuật**: PCA, LDA, ICA, Gabor Wavelet
- **Ứng dụng**: Nhận diện khuôn mặt trong điều kiện phức tạp

#### 3. Template / Statistical / Neural Networks Based
- **Template Matching**: Đối sánh trực tiếp các pixel/mẫu
- **Statistical**: PCA, DCT, LDA, LPP, ICA, Wavelet Gabor
- **Neural Networks**: CNN, Autoencoders, Siamese Networks
- **Deep Learning**: FaceNet, VGGFace, ArcFace

### Thuật toán sử dụng trong ứng dụng

#### Primary: Neural Networks (face_recognition library)
- Sử dụng dlib với CNN models
- Face encodings 128-dimensional
- Euclidean distance cho so sánh
- Độ chính xác cao, xử lý real-time

#### Fallback: Appearance-Based + Statistical (OpenCV)
- Haar Cascade cho face detection
- Thuật toán đa phương pháp:
  * Histogram Comparison (30%)
  * Template Matching (30%)
  * SSIM - Structural Similarity (25%)
  * Edge Detection (15%)
- PCA và LDA cho dimensionality reduction

### Confidence Levels
- **Very High**: 85%+ (chỉ face_recognition)
- **High**: 75-85% (face_recognition) / 55-70% (OpenCV)
- **Medium**: 65-75% (face_recognition) / 50-55% (OpenCV)
- **Low**: 55-65% (face_recognition) / 45-50% (OpenCV)

### Verification (1:1 Comparison)
- **face_recognition**: Ngưỡng 80% (very high confidence)
- **OpenCV**: Ngưỡng 60% (high confidence)
- Phương pháp: One-to-One Matching
- Ứng dụng: Xác thực danh tính chính xác cao

### Video Processing
- Xử lý theo frame với sampling thông minh
- Tối đa 20 frame có khuôn mặt per video
- Rate limiting: 5 FPS để tối ưu hiệu suất
- Bounding box và label real-time

## API Endpoints

### Recognition APIs
- `POST /recognize` - Nhận diện khuôn mặt từ ảnh
- `POST /recognize_video` - Nhận diện khuôn mặt từ video
- `POST /batch_recognize_images` - Xử lý batch nhiều ảnh
- `POST /batch_recognize_videos` - Xử lý batch nhiều video
- `POST /verify_face` - Xác minh danh tính khuôn mặt
- `POST /process_camera_frame` - Xử lý frame camera real-time

### Management APIs
- `POST /update_face_metadata/<name>` - Cập nhật metadata
- `GET /get_face_details/<name>` - Lấy thông tin chi tiết
- `GET /system_status` - Trạng thái hệ thống
- `GET /cleanup_temp` - Dọn dẹp file tạm
- `GET /refresh_metadata` - Làm mới metadata

### Debug APIs
- `GET /debug_faces` - Debug database
- `GET /test_recognition/<name>` - Test nhận diện

## Cải tiến gần đây

### 🎯 Tính năng mới
- ✅ **Real-time Camera**: Nhận diện khuôn mặt từ camera trực tiếp
- ✅ **Video Support**: Nhận diện khuôn mặt từ video
- ✅ **Batch Processing**: Xử lý nhiều file cùng lúc
- ✅ **Face Verification**: Xác minh danh tính 1:1
- ✅ **Enhanced Database**: Metadata phong phú
- ✅ **Advanced UI**: Giao diện tab với nhiều chức năng

### ⚡ Hiệu suất
- ✅ **Performance Monitoring**: Theo dõi thời gian xử lý
- ✅ **Memory Optimization**: Quản lý bộ nhớ thông minh
- ✅ **Image Optimization**: Resize ảnh lớn tự động
- ✅ **Auto Cleanup**: Dọn dẹp file tạm tự động

### 🎨 Độ chính xác
- ✅ **Multi-confidence Levels**: Phân cấp độ tin cậy
- ✅ **Advanced Algorithms**: Thuật toán đa phương pháp
- ✅ **Quality Assessment**: Đánh giá chất lượng ảnh
- ✅ **Smart Filtering**: Lọc khuôn mặt chất lượng cao

## Ứng dụng của công nghệ nhận diện khuôn mặt

Với tính chính xác cao, an toàn và tiện lợi, công nghệ nhận diện khuôn mặt được ứng dụng rộng rãi:

### 🔐 An ninh và xác thực
- **Xác thực danh tính**: Thay thế mật khẩu truyền thống cho điện thoại, máy tính
- **Kiểm soát truy cập**: Cửa ra vào, phòng server, khu vực bảo mật
- **Phát hiện gian lận**: Ngăn chặn tạo tài khoản giả, bảo vệ giao dịch tài chính
- **An ninh mạng**: Xác thực hai yếu tố, bảo vệ dữ liệu nhạy cảm

### 🏛️ An ninh công cộng
- **Quản lý sân bay**: Kiểm soát biên giới, boarding pass điện tử
- **Giám sát đám đông**: Phát hiện tội phạm, duy trì trật tự
- **Camera giám sát**: Theo dõi real-time tại các địa điểm công cộng
- **Nhận diện tội phạm**: So khớp với database tội phạm

### 🏥 Chăm sóc sức khỏe
- **Quản lý bệnh án**: Xác thực danh tính bệnh nhân
- **Theo dõi cảm xúc**: Phát hiện đau đớn, lo lắng của bệnh nhân
- **Bảo mật thông tin**: Kiểm soát truy cập dữ liệu y tế
- **Phát hiện bệnh nhân**: Tìm kiếm nhanh trong database

### 🛍️ Bán lẻ và thương mại
- **Cá nhân hóa trải nghiệm**: Nhận diện khách hàng thân thiết
- **Phân tích hành vi**: Theo dõi thói quen mua sắm
- **Bảo mật thanh toán**: Xác thực giao dịch tại POS
- **Đề xuất sản phẩm**: Phân tích sở thích dựa trên nhân diện

### 🚗 Giao thông và vận tải
- **Kiểm soát phương tiện**: Nhận diện tài xế, hành khách
- **Bảo mật kho bãi**: Kiểm soát truy cập khu vực logistics
- **Quản lý bến bãi**: Tự động hóa check-in/check-out

### 🏫 Giáo dục
- **Điểm danh tự động**: Theo dõi sự hiện diện của học sinh
- **Bảo mật trường học**: Kiểm soát truy cập khu vực
- **Quản lý thư viện**: Theo dõi mượn/trả sách
- **Bảo mật kỳ thi**: Ngăn chặn thi hộ

### 💼 Doanh nghiệp
- **Quản lý nhân sự**: Theo dõi giờ làm việc
- **Bảo mật văn phòng**: Kiểm soát truy cập tầng/lầu
- **Đào tạo nhân viên**: Theo dõi tham gia khóa học
- **Bảo mật hội nghị**: Kiểm soát danh sách tham gia

## Tác giả

**Quách Việt Tùng** - [GitHub](https://github.com/quachviettung)

## License

MIT License

## Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## Hỗ trợ

Nếu gặp vấn đề, hãy tạo issue trên GitHub repository.
