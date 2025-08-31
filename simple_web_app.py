#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng web đơn giản để demo giao diện nhận diện khuôn mặt
Simple Web App for Face Recognition UI Demo
"""

import os
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import face_recognition nếu có sẵn, nếu không thì fallback về OpenCV
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✓ Sử dụng thư viện face_recognition cho độ chính xác cao")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠ Không có face_recognition, sử dụng OpenCV cơ bản")

app = Flask(__name__)
app.secret_key = 'face_recognition_demo_key_2024'

# Cấu hình
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_FOLDER = 'known_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Tạo thư mục cần thiết
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

# Cơ sở dữ liệu khuôn mặt đã biết (đơn giản)
KNOWN_FACES_DB = {}
KNOWN_FACE_ENCODINGS = {}  # Lưu trữ encoding của khuôn mặt đã biết

def allowed_file(filename):
    """Kiểm tra file có được phép upload không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_known_faces():
    """Load cơ sở dữ liệu khuôn mặt đã biết"""
    # Thử load từ file JSON để lưu trữ bền vững
    json_path = os.path.join(KNOWN_FACES_FOLDER, 'known_faces.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                global KNOWN_FACES_DB
                KNOWN_FACES_DB = data
                
                # Tạo encoding cho các khuôn mặt đã biết
                if FACE_RECOGNITION_AVAILABLE:
                    create_face_encodings(data)
                
                return data
        except Exception as e:
            print(f"Lỗi khi load database: {e}")
    
    return KNOWN_FACES_DB

def create_face_encodings(known_faces):
    """Tạo encoding cho các khuôn mặt đã biết"""
    global KNOWN_FACE_ENCODINGS
    KNOWN_FACE_ENCODINGS = {}
    
    print("🔄 Đang tạo encoding cho các khuôn mặt đã biết...")
    
    for name, face_data in known_faces.items():
        if 'image_path' in face_data and os.path.exists(face_data['image_path']):
            try:
                # Load ảnh và tạo encoding
                image = face_recognition.load_image_file(face_data['image_path'])
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    # Lấy encoding đầu tiên (khuôn mặt chính)
                    KNOWN_FACE_ENCODINGS[name] = face_encodings[0]
                    print(f"✓ Đã tạo encoding cho {name}")
                else:
                    print(f"⚠ Không tìm thấy khuôn mặt trong ảnh của {name}")
                    
            except Exception as e:
                print(f"❌ Lỗi khi tạo encoding cho {name}: {e}")
    
    print(f"✅ Hoàn thành tạo encoding cho {len(KNOWN_FACE_ENCODINGS)} khuôn mặt")

def save_known_faces(known_faces):
    """Lưu cơ sở dữ liệu khuôn mặt"""
    global KNOWN_FACES_DB
    KNOWN_FACES_DB = known_faces
    
    # Lưu vào file JSON để bền vững
    json_path = os.path.join(KNOWN_FACES_FOLDER, 'known_faces.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(known_faces, f, ensure_ascii=False, indent=2)
        
        # Tạo lại encoding nếu có thay đổi
        if FACE_RECOGNITION_AVAILABLE:
            create_face_encodings(known_faces)
            
    except Exception as e:
        print(f"Lỗi khi lưu database: {e}")

def detect_faces_advanced(image_path, known_faces_db=None):
    """Phát hiện và nhận dạng khuôn mặt sử dụng face_recognition"""
    if not FACE_RECOGNITION_AVAILABLE:
        print("⚠ face_recognition không khả dụng, fallback về OpenCV")
        return detect_faces_simple(image_path, known_faces_db)
    
    try:
        print(f"\n🔍 Bắt đầu nhận dạng khuôn mặt từ: {image_path}")
        
        # Load ảnh
        image = face_recognition.load_image_file(image_path)
        
        # Tìm vị trí khuôn mặt
        face_locations = face_recognition.face_locations(image)
        print(f"📍 Phát hiện {len(face_locations)} khuôn mặt")
        
        if len(face_locations) == 0:
            print("❌ Không tìm thấy khuôn mặt nào")
            return [], 0
        
        # Tạo encoding cho khuôn mặt trong ảnh
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"🔐 Đã tạo {len(face_encodings)} face encoding")
        
        # Sử dụng known_faces_db được truyền vào hoặc KNOWN_FACES_DB global
        if known_faces_db is None:
            known_faces_db = KNOWN_FACES_DB
        
        results = []
        
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            print(f"\n--- Xử lý khuôn mặt {i+1} ---")
            
            # Chuyển đổi tọa độ
            top, right, bottom, left = face_location
            
            # So sánh với các khuôn mặt đã biết
            matches = []
            best_match = None
            best_similarity = 0.0
            
            print(f"🔍 So sánh với {len(KNOWN_FACE_ENCODINGS)} khuôn mặt đã biết...")
            
            for name, known_encoding in KNOWN_FACE_ENCODINGS.items():
                try:
                    # Tính khoảng cách giữa 2 encoding (càng nhỏ càng giống nhau)
                    face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    
                    # Chuyển đổi thành độ tương đồng (0-1, càng cao càng giống)
                    similarity = 1.0 - face_distance
                    
                    print(f"  📊 {name}: distance={face_distance:.4f}, similarity={similarity:.4f}")
                    
                    # Ngưỡng nhận dạng: similarity > 0.6 (có thể điều chỉnh)
                    if similarity > 0.6:
                        matches.append((name, float(similarity)))
                        print(f"    ✅ {name} được nhận dạng với similarity: {similarity:.4f}")
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (name, float(similarity))
                            print(f"    🎯 {name} trở thành best_match!")
                    else:
                        print(f"    ❌ {name} không đủ tương đồng")
                        
                except Exception as e:
                    print(f"    ⚠️ Lỗi khi so sánh với {name}: {e}")
            
            # Sắp xếp theo độ tương đồng
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Tạo kết quả
            result = {
                'location': {
                    'top': int(top), 
                    'right': int(right), 
                    'bottom': int(bottom), 
                    'left': int(left)
                },
                'matches': matches,
                'best_match': best_match,
                'face_index': i + 1
            }
            
            results.append(result)
            
            if best_match:
                name, similarity = best_match
                print(f"🎉 Khuôn mặt {i+1} được nhận dạng: {name} (similarity: {similarity:.4f})")
            else:
                print(f"❓ Khuôn mặt {i+1}: Không xác định được")
        
        print(f"\n📋 KẾT QUẢ CUỐI CÙNG:")
        print(f"  - Tổng khuôn mặt: {len(face_locations)}")
        print(f"  - Khuôn mặt được nhận dạng: {len([r for r in results if r['best_match']])}")
        print(f"  - Khuôn mặt không xác định: {len([r for r in results if not r['best_match']])}")
        
        return results, len(face_locations)
        
    except Exception as e:
        print(f"❌ Lỗi khi nhận dạng khuôn mặt: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

def detect_faces_simple(image_path, known_faces_db=None):
    """Phát hiện khuôn mặt đơn giản sử dụng OpenCV (cải tiến)"""
    try:
        print(f"\n🔍 Bắt đầu nhận dạng khuôn mặt từ: {image_path}")
        
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Không thể load ảnh")
            return [], 0
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sử dụng Haar Cascade với tham số tối ưu
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Tham số cân bằng để phát hiện khuôn mặt chính xác hơn
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,      # Tăng độ chính xác
            minNeighbors=6,       # Tăng độ tin cậy
            minSize=(50, 50),     # Kích thước tối thiểu hợp lý
            maxSize=(300, 300),   # Kích thước tối đa hợp lý
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"📍 Phát hiện {len(faces)} khuôn mặt ban đầu")
        
        # Lọc khuôn mặt với tiêu chí nghiêm ngặt hơn
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Kiểm tra kích thước hợp lý
            if w < 50 or h < 50 or w > 300 or h > 300:
                continue
                
            # Tỷ lệ khung hình phải gần vuông (0.8-1.2) - nghiêm ngặt hơn
            aspect_ratio = w / h
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue
                
            # Loại bỏ khuôn mặt ở viền ảnh (cách viền ít nhất 20px)
            if x < 20 or y < 20 or x + w > gray.shape[1] - 20 or y + h > gray.shape[0] - 20:
                continue
                
            # Kiểm tra chất lượng khuôn mặt (độ tương phản)
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
                
            # Tính độ tương phản của khuôn mặt
            contrast = np.std(face_roi)
            if contrast < 20:  # Tăng ngưỡng độ tương phản
                continue
                
            filtered_faces.append((x, y, w, h))
        
        faces = filtered_faces
        print(f"✅ Sau khi lọc: {len(faces)} khuôn mặt hợp lệ")
        
        # CHỈ XỬ LÝ 1 KHUÔN MẶT DUY NHẤT - lấy khuôn mặt lớn nhất (rõ ràng nhất)
        if len(faces) > 1:
            # Sắp xếp theo diện tích khuôn mặt (lớn nhất = rõ ràng nhất)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            faces = faces[:1]  # Chỉ giữ 1 khuôn mặt lớn nhất
            print(f"🎯 Chọn khuôn mặt lớn nhất để xử lý")
        elif len(faces) == 0:
            print("⚠ Không phát hiện được khuôn mặt nào sau khi lọc")
            return [], 0
        else:
            print(f"✓ Phát hiện 1 khuôn mặt duy nhất")
        
        # Sử dụng known_faces_db được truyền vào hoặc KNOWN_FACES_DB global
        if known_faces_db is None:
            known_faces_db = KNOWN_FACES_DB
        
        results = []
        for (x, y, w, h) in faces:
            print(f"\n--- Xử lý khuôn mặt chính ---")
            
            # Tính toán vị trí
            top, left = int(y), int(x)
            bottom, right = int(y + h), int(x + w)
            
            # Lấy ảnh khuôn mặt hiện tại để so sánh
            current_face_roi = gray[y:y+h, x:x+w]
            
            # Tìm khuôn mặt có độ tương đồng cao nhất
            matches = []
            best_match = None
            best_similarity = 0.0
            
            print(f"🔍 So sánh với {len(known_faces_db)} khuôn mặt đã biết...")
            
            for name, face_data in known_faces_db.items():
                if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                    try:
                        print(f"  📸 Đang xử lý: {name}")
                        
                        # Load ảnh khuôn mặt đã đăng ký
                        registered_image = cv2.imread(face_data['image_path'])
                        if registered_image is not None:
                            # Chuyển sang grayscale
                            registered_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)
                            
                            # Tìm khuôn mặt trong ảnh đã đăng ký với tham số nghiêm ngặt
                            registered_faces = face_cascade.detectMultiScale(
                                registered_gray, 
                                scaleFactor=1.05,
                                minNeighbors=8,
                                minSize=(60, 60),
                                maxSize=(200, 200)
                            )
                            
                            if len(registered_faces) > 0:
                                # Lấy khuôn mặt đầu tiên từ ảnh đã đăng ký
                                rx, ry, rw, rh = registered_faces[0]
                                registered_face_roi = registered_gray[ry:ry+rh, rx:rx+rw]
                                
                                # Resize để so sánh cùng kích thước (128x128 để tăng độ chính xác)
                                target_size = (128, 128)
                                registered_face_resized = cv2.resize(registered_face_roi, target_size)
                                current_face_resized = cv2.resize(current_face_roi, target_size)
                                
                                # Tính độ tương đồng với nhiều phương pháp
                                similarity = calculate_face_similarity_improved(current_face_resized, registered_face_resized)
                                
                                print(f"    📊 Độ tương đồng: {similarity:.4f}")
                                
                                # Ngưỡng nhận dạng cao hơn để tránh nhận dạng sai
                                if similarity > 0.45:  # Tăng ngưỡng lên 0.45
                                    matches.append((name, float(similarity)))
                                    print(f"    ✅ {name} được nhận dạng!")
                                    
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = (name, float(similarity))
                                        print(f"    🎯 {name} trở thành best_match!")
                                else:
                                    print(f"    ❌ {name} không đủ tương đồng")
                                    
                            else:
                                print(f"    ⚠️ Không tìm thấy khuôn mặt trong ảnh đã đăng ký")
                                    
                    except Exception as e:
                        print(f"    ❌ Lỗi khi xử lý ảnh của {name}: {e}")
            
            # Sắp xếp theo độ tương đồng
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Chỉ trả về kết quả nếu có match thực sự với độ tin cậy đủ cao
            if best_match and best_similarity > 0.45:
                results.append({
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'matches': matches[:1],  # Chỉ giữ 1 match tốt nhất
                    'best_match': best_match
                })
                print(f"🎉 Kết quả cuối cùng: {best_match[0]} (similarity: {best_match[1]:.4f})")
            else:
                # Nếu không có match nào, vẫn trả về khuôn mặt nhưng không nhận diện được
                results.append({
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'matches': [],
                    'best_match': None
                })
                print(f"❓ Không xác định được khuôn mặt này")
        
        print(f"\n📋 KẾT QUẢ CUỐI CÙNG:")
        print(f"  - Tổng khuôn mặt: {len(faces)}")
        print(f"  - Khuôn mặt được nhận dạng: {len([r for r in results if r['best_match']])}")
        print(f"  - Khuôn mặt không xác định: {len([r for r in results if not r['best_match']])}")
        
        return results, int(len(faces))
        
    except Exception as e:
        print(f"❌ Lỗi khi phát hiện khuôn mặt: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

def calculate_face_similarity_improved(face1, face2):
    """Tính độ tương đồng giữa hai khuôn mặt với thuật toán cải tiến"""
    try:
        # Đảm bảo kích thước ảnh giống nhau
        if face1.shape != face2.shape:
            face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
        
        # Phương pháp 1: Histogram comparison (cải tiến)
        hist1 = cv2.calcHist([face1], [0], None, [128], [0, 256])  # Tăng bins để chính xác hơn
        hist2 = cv2.calcHist([face2], [0], None, [128], [0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_similarity = (hist_similarity + 1) / 2  # Chuyển từ [-1, 1] sang [0, 1]
        
        # Phương pháp 2: Template matching (cải tiến)
        if face1.shape[0] >= face2.shape[0] and face1.shape[1] >= face2.shape[1]:
            result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
            template_similarity = np.max(result)
        else:
            # Nếu face1 nhỏ hơn face2, swap để so sánh
            result = cv2.matchTemplate(face2, face1, cv2.TM_CCOEFF_NORMED)
            template_similarity = np.max(result)
        
        # Phương pháp 3: Structural Similarity Index (SSIM) - cải tiến
        # Tính độ tương đồng cấu trúc
        face1_norm = face1.astype(float) / 255.0
        face2_norm = face2.astype(float) / 255.0
        
        # Tính SSIM đơn giản
        mu1 = np.mean(face1_norm)
        mu2 = np.mean(face2_norm)
        sigma1 = np.std(face1_norm)
        sigma2 = np.std(face2_norm)
        sigma12 = np.mean((face1_norm - mu1) * (face2_norm - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        ssim = max(0, min(1, ssim))  # Đảm bảo trong khoảng [0, 1]
        
        # Phương pháp 4: Edge detection comparison
        # Phát hiện cạnh và so sánh
        edges1 = cv2.Canny(face1, 50, 150)
        edges2 = cv2.Canny(face2, 50, 150)
        
        edge_similarity = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
        edge_similarity = np.max(edge_similarity)
        
        # Kết hợp các phương pháp với trọng số tối ưu
        combined_similarity = (
            0.30 * hist_similarity +     # Histogram: 30%
            0.30 * template_similarity + # Template: 30%
            0.25 * ssim +               # SSIM: 25%
            0.15 * edge_similarity      # Edge: 15%
        )
        
        # Áp dụng penalty để tăng độ chính xác
        if combined_similarity < 0.4:
            combined_similarity *= 0.7  # Giảm độ tương đồng thấp
        elif combined_similarity < 0.6:
            combined_similarity *= 0.85  # Giảm độ tương đồng trung bình
        
        return max(0.0, min(1.0, combined_similarity))
        
    except Exception as e:
        print(f"❌ Lỗi khi tính độ tương đồng: {e}")
        return 0.0  # Trả về 0 nếu có lỗi

def calculate_face_similarity(face1, face2):
    """Tính độ tương đồng giữa hai khuôn mặt dựa trên nhiều phương pháp cải tiến (legacy)"""
    return calculate_face_similarity_improved(face1, face2)

@app.route('/')
def index():
    """Trang chủ"""
    known_faces = load_known_faces()
    return render_template('index.html', known_faces=known_faces)

@app.route('/register', methods=['GET', 'POST'])
def register_face():
    """Đăng ký khuôn mặt mới"""
    if request.method == 'POST':
        if 'name' not in request.form or 'image' not in request.files:
            flash('Vui lòng nhập tên và chọn ảnh!', 'error')
            return redirect(request.url)
        
        name = request.form['name'].strip()
        image_file = request.files['image']
        
        if name == '' or image_file.filename == '':
            flash('Vui lòng nhập tên và chọn ảnh!', 'error')
            return redirect(request.url)
        
        if image_file and allowed_file(image_file.filename):
            # Lưu ảnh
            filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)
            
            # Kiểm tra xem có khuôn mặt trong ảnh không
            faces, count = detect_faces_advanced(image_path, load_known_faces())
            
            if count > 0:
                # Lưu vào cơ sở dữ liệu
                known_faces = load_known_faces()
                known_faces[name] = {
                    'image_path': image_path,
                    'registered_at': datetime.now().isoformat(),
                    'face_count': count
                }
                save_known_faces(known_faces)
                
                flash(f'Đăng ký khuôn mặt cho {name} thành công! Tìm thấy {count} khuôn mặt.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Không thể phát hiện khuôn mặt trong ảnh!', 'error')
                os.remove(image_path)  # Xóa ảnh không hợp lệ
        else:
            flash('File không được hỗ trợ!', 'error')
    
    return render_template('register.html')

@app.route('/recognize', methods=['POST'])
def recognize_face_api():
    """API nhận diện khuôn mặt"""
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh được upload'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Không có ảnh được chọn'}), 400
    
    if image_file and allowed_file(image_file.filename):
        # Lưu ảnh tạm thời
        filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        
        try:
            # Phát hiện khuôn mặt
            known_faces = load_known_faces()
            results, face_count = detect_faces_advanced(image_path, known_faces)
            
            # Xóa file tạm
            os.remove(image_path)
            
            return jsonify({
                'success': True,
                'results': results,
                'face_count': face_count
            })
            
        except Exception as e:
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File không được hỗ trợ'}), 400

@app.route('/delete_face/<name>')
def delete_face(name):
    """Xóa khuôn mặt đã đăng ký"""
    known_faces = load_known_faces()
    if name in known_faces:
        # Xóa file ảnh
        image_path = known_faces[name]['image_path']
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Xóa khỏi database
        del known_faces[name]
        save_known_faces(known_faces)
        
        flash(f'Đã xóa khuôn mặt của {name}!', 'success')
    else:
        flash(f'Không tìm thấy khuôn mặt của {name}!', 'error')
    
    return redirect(url_for('index'))

@app.route('/camera')
def camera():
    """Trang camera real-time"""
    return render_template('camera.html')

@app.route('/demo_faces')
def demo_faces():
    """Tạo dữ liệu demo"""
    known_faces = {
        'Nguyễn Văn A': {
            'image_path': 'demo/person1.jpg',
            'registered_at': datetime.now().isoformat(),
            'face_count': 1
        },
        'Trần Thị B': {
            'image_path': 'demo/person2.jpg',
            'registered_at': datetime.now().isoformat(),
            'face_count': 1
        },
        'Lê Văn C': {
            'image_path': 'demo/person3.jpg',
            'registered_at': datetime.now().isoformat(),
            'face_count': 1
        }
    }
    save_known_faces(known_faces)
    flash('Đã tạo dữ liệu demo!', 'success')
    return redirect(url_for('index'))

@app.route('/debug_faces')
def debug_faces():
    """Debug: Hiển thị thông tin database"""
    known_faces = load_known_faces()
    debug_info = {
        'total_faces': len(known_faces),
        'faces': known_faces,
        'global_db': KNOWN_FACES_DB
    }
    return jsonify(debug_info)

@app.route('/test_recognition/<name>')
def test_recognition(name):
    """Test nhận diện một khuôn mặt cụ thể"""
    known_faces = load_known_faces()
    if name not in known_faces:
        return jsonify({'error': f'Không tìm thấy khuôn mặt của {name}'}), 404
    
    face_data = known_faces[name]
    if 'image_path' not in face_data or not os.path.exists(face_data['image_path']):
        return jsonify({'error': f'Không tìm thấy ảnh của {name}'}), 404
    
    try:
        # Test nhận diện chính ảnh đã đăng ký
        results, face_count = detect_faces_advanced(face_data['image_path'], known_faces)
        
        return jsonify({
            'name': name,
            'image_path': face_data['image_path'],
            'results': results,
            'face_count': face_count,
            'message': f'Test nhận diện cho {name}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=== ỨNG DỤNG WEB NHẬN DIỆN KHUÔN MẶT ===")
    
    # Tạo dữ liệu demo khi khởi động nếu database trống
    known_faces = load_known_faces()
    if len(known_faces) == 0:
        print("Tạo dữ liệu demo...")
        demo_faces = {
            'Nguyễn Văn A': {
                'image_path': 'demo/person1.jpg',
                'registered_at': datetime.now().isoformat(),
                'face_count': 1
            },
            'Trần Thị B': {
                'image_path': 'demo/person2.jpg',
                'registered_at': datetime.now().isoformat(),
                'face_count': 1
            }
        }
        save_known_faces(demo_faces)
        print(f"Đã tạo {len(demo_faces)} khuôn mặt demo")
    else:
        # Tạo encoding cho các khuôn mặt hiện có (chỉ khi có face_recognition)
        if FACE_RECOGNITION_AVAILABLE:
            print("🔄 Tạo encoding cho các khuôn mặt hiện có...")
            create_face_encodings(known_faces)
        else:
            print("ℹ️ Sử dụng OpenCV với thuật toán cải tiến")
    
    print("Khởi động web server...")
    print("Truy cập: http://localhost:5000")
    if FACE_RECOGNITION_AVAILABLE:
        print("✅ Sử dụng face_recognition library cho độ chính xác cao")
    else:
        print("⚠ Sử dụng OpenCV cơ bản (độ chính xác thấp)")
        print("Để có tính năng đầy đủ, cần cài đặt face_recognition library")
    app.run(debug=True, host='0.0.0.0', port=5000)
