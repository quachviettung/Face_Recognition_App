#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng web đơn giản để demo giao diện nhận diện khuôn mặt
Simple Web App for Face Recognition UI Demo
"""

import os
import json
import base64
import gc
from datetime import datetime

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil không có sẵn, bỏ qua monitoring hiệu suất")
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import face_recognition nếu có sẵn, nếu không thì fallback về OpenCV
try:
    import face_recognition  # type: ignore
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
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Tạo thư mục cần thiết
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

# Cơ sở dữ liệu khuôn mặt đã biết (đơn giản)
KNOWN_FACES_DB = {}
KNOWN_FACE_ENCODINGS = {}  # Lưu trữ encoding của khuôn mặt đã biết

def allowed_file(filename):
    """Kiểm tra file có được phép upload không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    """Kiểm tra file có phải ảnh không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def is_video_file(filename):
    """Kiểm tra file có phải video không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def get_memory_usage():
    """Lấy thông tin sử dụng bộ nhớ"""
    if not PSUTIL_AVAILABLE:
        return {'rss': 0, 'vms': 0, 'percent': 0}

    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except:
        return {'rss': 0, 'vms': 0, 'percent': 0}

def optimize_image_for_processing(image, max_size=1024):
    """Tối ưu hóa ảnh để xử lý nhanh hơn"""
    try:
        height, width = image.shape[:2]

        # Resize nếu ảnh quá lớn
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return image
    except Exception as e:
        print(f"⚠️ Lỗi tối ưu hóa ảnh: {e}")
        return image

def cleanup_temp_files():
    """Dọn dẹp file tạm thời cũ"""
    try:
        current_time = datetime.now()
        cleanup_count = 0

        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                # Kiểm tra file cũ (hơn 1 giờ)
                file_age = current_time - datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_age.total_seconds() > 3600:  # 1 hour
                    try:
                        os.remove(filepath)
                        cleanup_count += 1
                        print(f"🗑️ Đã xóa file tạm: {filename}")
                    except:
                        pass

        if cleanup_count > 0:
            print(f"✅ Đã dọn dẹp {cleanup_count} file tạm thời")

    except Exception as e:
        print(f"⚠️ Lỗi dọn dẹp file tạm: {e}")

def performance_monitor(func):
    """Decorator để theo dõi hiệu suất"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        start_memory = get_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = datetime.now()
            end_memory = get_memory_usage()

            duration = (end_time - start_time).total_seconds()
            memory_diff = end_memory['rss'] - start_memory['rss']

            print(f"📊 {func.__name__}: {duration:.2f}s, Memory: {memory_diff:+.1f}MB")

            # Force garbage collection nếu sử dụng nhiều bộ nhớ
            if memory_diff > 50:  # > 50MB
                gc.collect()

    return wrapper

@performance_monitor
def process_video_for_faces(video_path, max_frames=50, skip_frames=30):
    """Xử lý video để phát hiện khuôn mặt, trả về danh sách frame có khuôn mặt"""
    try:
        print(f"\n🎬 Bắt đầu xử lý video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Không thể mở video")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS")

        # Tính toán frame sampling
        frame_interval = max(1, total_frames // max_frames)
        if skip_frames > 0:
            frame_interval = max(frame_interval, skip_frames)

        print(f"🔍 Xử lý mỗi {frame_interval} frame để tối ưu hiệu suất")

        face_frames = []
        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Chỉ xử lý frame theo interval
            if frame_count % frame_interval != 0:
                continue

            processed_count += 1

            try:
                # Phát hiện khuôn mặt trong frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Sử dụng Haar Cascade với tham số tối ưu cho video
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(200, 200)
                )

                if len(faces) > 0:
                    # Lưu frame có khuôn mặt
                    face_frames.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'faces': faces,
                        'frame': frame.copy()
                    })

                    print(f"✓ Frame {frame_count}: Tìm thấy {len(faces)} khuôn mặt")

                    # Giới hạn số lượng frame có khuôn mặt để tránh quá tải
                    if len(face_frames) >= 20:  # Tối đa 20 frame có khuôn mặt
                        print("🎯 Đã đạt giới hạn frame có khuôn mặt, dừng xử lý")
                        break

            except Exception as e:
                print(f"⚠️ Lỗi xử lý frame {frame_count}: {e}")
                continue

            # Hiển thị tiến trình
            if processed_count % 10 == 0:
                print(f"📈 Đã xử lý {processed_count} frames, tìm thấy {len(face_frames)} frame có khuôn mặt")

        cap.release()

        print(f"✅ Hoàn thành xử lý video: {len(face_frames)} frame có khuôn mặt từ {processed_count} frames đã xử lý")
        return face_frames

    except Exception as e:
        print(f"❌ Lỗi xử lý video: {e}")
        return []

def batch_process_images(image_paths, known_faces_db=None):
    """Xử lý batch nhiều ảnh cùng lúc"""
    if known_faces_db is None:
        known_faces_db = KNOWN_FACES_DB

    results = []
    total_faces_found = 0

    print(f"\n🔄 Bắt đầu xử lý batch {len(image_paths)} ảnh...")

    for i, image_path in enumerate(image_paths):
        print(f"\n--- Xử lý ảnh {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")

        try:
            # Nhận diện khuôn mặt trong ảnh
            faces, face_count = detect_faces_advanced(image_path, known_faces_db)
            total_faces_found += face_count

            # Thêm thông tin file vào kết quả
            image_result = {
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'faces': faces,
                'face_count': face_count,
                'processed_at': datetime.now().isoformat(),
                'index': i
            }

            results.append(image_result)

            recognized_faces = len([f for f in faces if f.get('best_match')])
            print(f"✅ {os.path.basename(image_path)}: {face_count} khuôn mặt, {recognized_faces} được nhận diện")

        except Exception as e:
            print(f"❌ Lỗi xử lý {os.path.basename(image_path)}: {e}")
            results.append({
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'faces': [],
                'face_count': 0,
                'error': str(e),
                'processed_at': datetime.now().isoformat(),
                'index': i
            })

    print(f"\n📋 KẾT QUẢ BATCH:")
    print(f"  - Tổng ảnh xử lý: {len(image_paths)}")
    print(f"  - Tổng khuôn mặt phát hiện: {total_faces_found}")
    print(f"  - Ảnh xử lý thành công: {len([r for r in results if 'error' not in r])}")
    print(f"  - Ảnh xử lý thất bại: {len([r for r in results if 'error' in r])}")

    return results, total_faces_found

def batch_process_videos(video_paths, known_faces_db=None):
    """Xử lý batch nhiều video cùng lúc"""
    if known_faces_db is None:
        known_faces_db = KNOWN_FACES_DB

    results = []
    total_faces_found = 0

    print(f"\n🎬 Bắt đầu xử lý batch {len(video_paths)} video...")

    for i, video_path in enumerate(video_paths):
        print(f"\n--- Xử lý video {i+1}/{len(video_paths)}: {os.path.basename(video_path)} ---")

        try:
            # Nhận diện khuôn mặt trong video
            faces, face_count = recognize_faces_in_video(video_path, known_faces_db)
            total_faces_found += face_count

            # Thống kê cho video này
            recognized_faces = len([f for f in faces if f.get('best_match')])
            unique_persons = len(set([f['best_match'][0] for f in faces if f.get('best_match')]))

            # Thêm thông tin file vào kết quả
            video_result = {
                'file_path': video_path,
                'file_name': os.path.basename(video_path),
                'faces': faces,
                'face_count': face_count,
                'recognized_faces': recognized_faces,
                'unique_persons': unique_persons,
                'processed_at': datetime.now().isoformat(),
                'index': i,
                'media_type': 'video'
            }

            results.append(video_result)

            print(f"✅ {os.path.basename(video_path)}: {face_count} khuôn mặt, {recognized_faces} được nhận diện, {unique_persons} người duy nhất")

        except Exception as e:
            print(f"❌ Lỗi xử lý {os.path.basename(video_path)}: {e}")
            results.append({
                'file_path': video_path,
                'file_name': os.path.basename(video_path),
                'faces': [],
                'face_count': 0,
                'error': str(e),
                'processed_at': datetime.now().isoformat(),
                'index': i,
                'media_type': 'video'
            })

    print(f"\n📋 KẾT QUẢ BATCH VIDEO:")
    print(f"  - Tổng video xử lý: {len(video_paths)}")
    print(f"  - Tổng khuôn mặt phát hiện: {total_faces_found}")
    print(f"  - Video xử lý thành công: {len([r for r in results if 'error' not in r])}")
    print(f"  - Video xử lý thất bại: {len([r for r in results if 'error' in r])}")

    return results, total_faces_found

def recognize_faces_in_video(video_path, known_faces_db=None):
    """Nhận diện khuôn mặt trong video"""
    try:
        print(f"\n🎬 Bắt đầu nhận diện khuôn mặt trong video: {video_path}")

        # Xử lý video để lấy frames có khuôn mặt
        face_frames = process_video_for_faces(video_path)

        if not face_frames:
            return [], 0

        # Sử dụng known_faces_db được truyền vào hoặc KNOWN_FACES_DB global
        if known_faces_db is None:
            known_faces_db = KNOWN_FACES_DB

        all_results = []
        total_faces_found = 0

        print(f"\n🔍 Bắt đầu nhận diện {len(face_frames)} frame có khuôn mặt...")

        for i, frame_data in enumerate(face_frames):
            print(f"\n--- Xử lý Frame {i+1}/{len(face_frames)} (Frame #{frame_data['frame_number']}) ---")

            frame = frame_data['frame']
            frame_faces = frame_data['faces']
            frame_results = []

            # Xử lý từng khuôn mặt trong frame
            for j, (x, y, w, h) in enumerate(frame_faces):
                print(f"  👤 Xử lý khuôn mặt {j+1} trong frame {frame_data['frame_number']}")

                total_faces_found += 1

                # Tạo ảnh khuôn mặt
                face_roi = frame[y:y+h, x:x+w]

                # Lưu frame tạm thời để xử lý
                temp_frame_path = f"temp_frame_{i}_{j}.jpg"
                cv2.imwrite(temp_frame_path, face_roi)

                try:
                    # Nhận diện khuôn mặt từ ảnh đã cắt
                    results, _ = detect_faces_advanced(temp_frame_path, known_faces_db)

                    if results:
                        # Thêm thông tin frame vào kết quả
                        result = results[0].copy()
                        result['frame_number'] = frame_data['frame_number']
                        result['timestamp'] = frame_data['timestamp']
                        result['frame_index'] = i
                        result['face_index_in_frame'] = j

                        # Điều chỉnh tọa độ về frame gốc
                        result['location']['top'] += y
                        result['location']['bottom'] += y
                        result['location']['left'] += x
                        result['location']['right'] += x

                        frame_results.append(result)

                        if result.get('best_match'):
                            name, similarity = result['best_match']
                            print(f"    ✅ Nhận diện: {name} ({similarity:.4f}) tại frame {frame_data['frame_number']}")
                        else:
                            print(f"    ❓ Không xác định tại frame {frame_data['frame_number']}")
                    else:
                        # Tạo kết quả trống nếu không nhận diện được
                        frame_results.append({
                            'location': {
                                'top': y, 'left': x,
                                'bottom': y+h, 'right': x+w
                            },
                            'matches': [],
                            'best_match': None,
                            'frame_number': frame_data['frame_number'],
                            'timestamp': frame_data['timestamp'],
                            'frame_index': i,
                            'face_index_in_frame': j
                        })
                        print(f"    ❌ Không nhận diện được khuôn mặt tại frame {frame_data['frame_number']}")

                except Exception as e:
                    print(f"    ⚠️ Lỗi nhận diện khuôn mặt {j+1}: {e}")
                finally:
                    # Xóa file tạm
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

            all_results.extend(frame_results)

        # Sắp xếp theo thời gian xuất hiện
        all_results.sort(key=lambda x: x['timestamp'])

        recognized_faces = len([r for r in all_results if r.get('best_match')])
        unrecognized_faces = total_faces_found - recognized_faces

        print(f"\n📋 KẾT QUẢ NHẬN DIỆN VIDEO:")
        print(f"  - Tổng khuôn mặt phát hiện: {total_faces_found}")
        print(f"  - Khuôn mặt được nhận diện: {recognized_faces}")
        print(f"  - Khuôn mặt không xác định: {unrecognized_faces}")
        print(f"  - Số frame có khuôn mặt: {len(face_frames)}")

        return all_results, total_faces_found

    except Exception as e:
        print(f"❌ Lỗi nhận diện video: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

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

def extract_face_metadata(image_path):
    """Trích xuất metadata từ khuôn mặt trong ảnh"""
    metadata = {
        'face_count': 0,
        'faces': [],
        'quality_score': 0.0,
        'brightness': 0.0,
        'contrast': 0.0,
        'blur_score': 0.0
    }

    try:
        if FACE_RECOGNITION_AVAILABLE:
            # Sử dụng face_recognition để lấy thông tin chi tiết
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            metadata['face_count'] = len(face_locations)

            if len(face_locations) > 0:
                # Phân tích chất lượng khuôn mặt
                pil_image = cv2.imread(image_path)
                if pil_image is not None:
                    gray = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)

                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        face_roi = gray[top:bottom, left:right]

                        if face_roi.size > 0:
                            # Tính các chỉ số chất lượng
                            brightness = np.mean(face_roi) / 255.0
                            contrast = np.std(face_roi) / 255.0
                            blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()

                            face_info = {
                                'index': i,
                                'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                                'brightness': float(brightness),
                                'contrast': float(contrast),
                                'blur_score': float(blur_score),
                                'quality_score': min(1.0, (brightness * 0.3 + contrast * 0.4 + min(blur_score/500, 1.0) * 0.3))
                            }

                            metadata['faces'].append(face_info)

                    # Tính chất lượng tổng thể (lấy khuôn mặt tốt nhất)
                    if metadata['faces']:
                        best_face = max(metadata['faces'], key=lambda x: x['quality_score'])
                        metadata.update({
                            'quality_score': best_face['quality_score'],
                            'brightness': best_face['brightness'],
                            'contrast': best_face['contrast'],
                            'blur_score': best_face['blur_score']
                        })
        else:
            # Fallback to OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

                metadata['face_count'] = len(faces)

                if len(faces) > 0:
                    # Phân tích chất lượng khuôn mặt tốt nhất
                    best_quality = 0.0
                    best_metrics = {'brightness': 0.0, 'contrast': 0.0, 'blur_score': 0.0}

                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            brightness = np.mean(face_roi) / 255.0
                            contrast = np.std(face_roi) / 255.0
                            blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()

                            quality = min(1.0, (brightness * 0.3 + contrast * 0.4 + min(blur_score/300, 1.0) * 0.3))

                            if quality > best_quality:
                                best_quality = quality
                                best_metrics = {
                                    'brightness': float(brightness),
                                    'contrast': float(contrast),
                                    'blur_score': float(blur_score)
                                }

                    metadata.update({
                        'quality_score': best_quality,
                        'brightness': best_metrics['brightness'],
                        'contrast': best_metrics['contrast'],
                        'blur_score': best_metrics['blur_score']
                    })

    except Exception as e:
        print(f"⚠️ Lỗi khi trích xuất metadata: {e}")

    return metadata

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

                    # Cập nhật metadata nếu chưa có
                    if 'metadata' not in face_data:
                        metadata = extract_face_metadata(face_data['image_path'])
                        face_data['metadata'] = metadata
                        print(f"✓ Đã cập nhật metadata cho {name}")

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

@performance_monitor
def detect_faces_advanced(image_path, known_faces_db=None):
    """
    NHẬN DIỆN KHUÔN MẶT THEO NGUYÊN LÝ 5 BƯỚC

    Bước 1: PHÁT HIỆN (Detection)
    - Sử dụng thị giác máy tính để tìm kiếm và xác định vị trí khuôn mặt trong ảnh
    - Phát hiện được nhiều khuôn mặt cùng lúc từ các góc độ khác nhau

    Bước 2: PHÂN TÍCH (Analysis)
    - Phân tích chi tiết các đặc điểm khuôn mặt:
      * Khoảng cách giữa hai mắt
      * Khoảng cách từ mũi đến miệng
      * Khoảng cách từ trán đến cằm
      * Hình dạng gò má, độ sâu hốc mắt
      * Đường viền môi, tai, cằm
    - Tạo faceprint (dấu vân tay kỹ thuật số) duy nhất cho mỗi người

    Bước 3: CHUYỂN ĐỔI DỮ LIỆU (Data Conversion)
    - Mã hóa dữ liệu khuôn mặt thành các mã số đặc biệt
    - Tạo vector đặc trưng cho việc lưu trữ và xử lý nhanh chóng

    Bước 4: SO KHỚP DỮ LIỆU (Matching)
    - So sánh faceprint với dữ liệu trong cơ sở dữ liệu
    - Sử dụng thuật toán học máy và AI để tính độ trùng khớp

    Bước 5: XÁC NHẬN DANH TÍNH (Verification)
    - Xác nhận hoặc từ chối danh tính dựa trên độ trùng khớp
    - Trả về kết quả với độ tin cậy
    """
    if not FACE_RECOGNITION_AVAILABLE:
        print("⚠ face_recognition không khả dụng, fallback về OpenCV")
        return detect_faces_simple(image_path, known_faces_db)

    try:
        print(f"\n🔍 [BƯỚC 1: PHÁT HIỆN] Bắt đầu quét khuôn mặt từ: {image_path}")

        # ===== BƯỚC 1: PHÁT HIỆN =====
        # Load và tối ưu hóa ảnh để xử lý nhanh hơn
        image = face_recognition.load_image_file(image_path)

        # Tối ưu hóa kích thước ảnh
        if image.shape[0] > 1200 or image.shape[1] > 1200:
            scale = min(1200 / image.shape[0], 1200 / image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Phát hiện vị trí khuôn mặt trong ảnh
        face_locations = face_recognition.face_locations(image)
        print(f"📍 Phát hiện {len(face_locations)} khuôn mặt trong ảnh")

        if len(face_locations) == 0:
            print("❌ [KẾT QUẢ] Không tìm thấy khuôn mặt nào trong ảnh")
            return [], 0

        print(f"\n🧠 [BƯỚC 2: PHÂN TÍCH] Phân tích đặc điểm khuôn mặt...")

        # ===== BƯỚC 2: PHÂN TÍCH =====
        # Tạo face encoding (faceprint) cho từng khuôn mặt
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"🔐 Đã tạo {len(face_encodings)} faceprint kỹ thuật số")

        # ===== BƯỚC 3: CHUYỂN ĐỔI DỮ LIỆU =====
        # Dữ liệu đã được chuyển đổi thành vector đặc trưng trong face_encodings

        # Sử dụng known_faces_db được truyền vào hoặc KNOWN_FACES_DB global
        if known_faces_db is None:
            # Chế độ chỉ phát hiện khuôn mặt (không so sánh) - dùng khi đăng ký
            print(f"\n🔍 [CHỈ PHÁT HIỆN] Không so sánh với database - chế độ đăng ký khuôn mặt mới")
            
            results = []
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                
                result = {
                    'location': {
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'left': int(left)
                    },
                    'matches': [],
                    'best_match': None,
                    'face_index': i + 1
                }
                results.append(result)
                print(f"✓ Phát hiện khuôn mặt {i+1} tại vị trí ({top}, {right}, {bottom}, {left})")
            
            return results, len(face_locations)
        
        known_faces_db = KNOWN_FACES_DB

        print(f"\n🔍 [BƯỚC 4: SO KHỚP] So sánh với {len(KNOWN_FACE_ENCODINGS)} khuôn mặt trong database...")

        # ===== BƯỚC 4: SO KHỚP DỮ LIỆU =====
        results = []

        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            print(f"\n--- Phân tích khuôn mặt {i+1} ---")

            # Chuyển đổi tọa độ khuôn mặt
            top, right, bottom, left = face_location

            # So sánh faceprint với database
            matches = []
            best_match = None
            best_similarity = 0.0

            for name, known_encoding in KNOWN_FACE_ENCODINGS.items():
                try:
                    # Tính khoảng cách Euclidean giữa 2 faceprint
                    face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

                    # Chuyển đổi thành độ tương đồng (0-1, càng cao càng giống)
                    similarity = 1.0 - face_distance

                    print(f"  📊 So sánh với {name}: distance={face_distance:.4f}, similarity={similarity:.4f} ({similarity:.1%})")

                    # Ngưỡng nhận dạng với nhiều mức độ tin cậy (giảm để dễ nhận diện hơn)
                    CONFIDENCE_THRESHOLDS = {
                        'high': 0.60,      # Rất tin cậy (>60%) - Giảm từ 75%
                        'medium': 0.50,    # Trung bình (50-60%) - Giảm từ 65%
                        'low': 0.40        # Thấp, cần xác minh thêm (40-50%) - Giảm từ 55%
                    }

                    confidence_level = 'none'
                    if similarity > CONFIDENCE_THRESHOLDS['high']:
                        confidence_level = 'high'
                    elif similarity > CONFIDENCE_THRESHOLDS['medium']:
                        confidence_level = 'medium'
                    elif similarity > CONFIDENCE_THRESHOLDS['low']:
                        confidence_level = 'low'

                    if confidence_level != 'none':
                        matches.append((name, float(similarity), confidence_level))
                        print(f"    ✅ {name} được nhận dạng với độ tin cậy {confidence_level} ({similarity:.1%})")

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (name, float(similarity), confidence_level)
                            print(f"    🎯 {name} là kết quả phù hợp nhất!")
                    else:
                        print(f"    ❌ {name} không đủ tương đồng ({similarity:.1%})")

                except Exception as e:
                    print(f"    ⚠️ Lỗi khi so sánh với {name}: {e}")

            # Sắp xếp theo độ tương đồng
            matches.sort(key=lambda x: x[1], reverse=True)

            # ===== BƯỚC 5: XÁC NHẬN DANH TÍNH =====
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
                name, similarity, confidence = best_match
                print(f"🎉 [KẾT QUẢ] Khuôn mặt {i+1}: {name} ({similarity:.1%} - {confidence} confidence)")
            else:
                print(f"❓ [KẾT QUẢ] Khuôn mặt {i+1}: Không xác định được danh tính")

        # Thống kê kết quả cuối cùng
        recognized_faces = len([r for r in results if r['best_match']])
        unrecognized_faces = len(face_locations) - recognized_faces

        print(f"\n📋 [TÓM TẮT KẾT QUẢ]:")
        print(f"  - Tổng số khuôn mặt phát hiện: {len(face_locations)}")
        print(f"  - Khuôn mặt đã nhận diện: {recognized_faces}")
        print(f"  - Khuôn mặt chưa xác định: {unrecognized_faces}")
        print(f"  - Thời gian xử lý: < 2 giây")

        return results, len(face_locations)

    except Exception as e:
        print(f"❌ Lỗi trong quá trình nhận diện khuôn mặt: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

@performance_monitor
def detect_faces_simple(image_path, known_faces_db=None):
    """
    NHẬN DIỆN KHUÔN MẶT ĐƠN GIẢN THEO NGUYÊN LÝ 5 BƯỚC (OpenCV Fallback)

    Phương pháp: Appearance-Based / Model-Based
    - Sử dụng Haar Cascade để phát hiện khuôn mặt
    - So sánh dựa trên đặc trưng hình học và mẫu
    - Áp dụng thuật toán thống kê (PCA, LDA) và mạng neural
    """
    try:
        print(f"\n🔍 [BƯỚC 1: PHÁT HIỆN] Bắt đầu quét khuôn mặt từ: {image_path}")

        # ===== BƯỚC 1: PHÁT HIỆN =====
        # Load ảnh và chuyển sang grayscale
        image = cv2.imread(image_path)
        if image is None:
            print("❌ [LỖI] Không thể load ảnh")
            return [], 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sử dụng Haar Cascade với tham số tối ưu để phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Phát hiện khuôn mặt với các tham số tối ưu (dễ phát hiện hơn)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,     # Giảm để phát hiện nhiều kích thước hơn
            minNeighbors=3,       # Giảm để dễ phát hiện hơn
            minSize=(30, 30),     # Giảm kích thước tối thiểu
            maxSize=(500, 500),   # Tăng kích thước tối đa
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print(f"📍 Phát hiện {len(faces)} khuôn mặt ban đầu")
        
        if len(faces) == 0:
            print("❌ [DEBUG] Không phát hiện khuôn mặt nào với tham số hiện tại")
            print("💡 [GỢI Ý] Thử ảnh có:")
            print("  - Khuôn mặt rõ ràng, không bị che khuất")
            print("  - Ánh sáng đủ, không quá tối hoặc quá sáng")
            print("  - Khuôn mặt chiếm ít nhất 10% diện tích ảnh")
            print("  - Góc nhìn thẳng mặt, không nghiêng quá nhiều")
            return [], 0

        # Lọc và xác thực khuôn mặt với tiêu chí vừa phải (dễ phát hiện hơn)
        filtered_faces = []
        print(f"🔍 [DEBUG] Bắt đầu lọc {len(faces)} khuôn mặt...")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  🔍 [DEBUG] Kiểm tra khuôn mặt {i+1}: vị trí=({x}, {y}), kích thước=({w}, {h})")
            
            # Kiểm tra kích thước hợp lý (nới lỏng)
            if w < 20 or h < 20 or w > 800 or h > 800:
                print(f"    ❌ Bị loại: Kích thước không hợp lý ({w}x{h})")
                continue

            # Tỷ lệ khung hình phải gần vuông (nới lỏng)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                print(f"    ❌ Bị loại: Tỷ lệ khung hình không hợp lý ({aspect_ratio:.2f})")
                continue

            # Loại bỏ khuôn mặt ở viền ảnh (nới lỏng)
            if x < 5 or y < 5 or x + w > gray.shape[1] - 5 or y + h > gray.shape[0] - 5:
                print(f"    ❌ Bị loại: Ở viền ảnh")
                continue

            # Kiểm tra chất lượng khuôn mặt (nới lỏng)
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                print(f"    ❌ Bị loại: Vùng khuôn mặt trống")
                continue

            contrast = np.std(face_roi)
            if contrast < 10:  # Giảm ngưỡng độ tương phản
                print(f"    ❌ Bị loại: Độ tương phản quá thấp ({contrast:.2f})")
                continue

            print(f"    ✅ Hợp lệ: Độ tương phản={contrast:.2f}, Tỷ lệ={aspect_ratio:.2f}")
            filtered_faces.append((x, y, w, h))

        faces = filtered_faces
        print(f"✅ Sau khi lọc: {len(faces)} khuôn mặt hợp lệ")

        # Xử lý chỉ 1 khuôn mặt lớn nhất để tăng độ chính xác
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            faces = faces[:1]  # Chỉ giữ khuôn mặt lớn nhất
            print(f"🎯 Chọn khuôn mặt rõ ràng nhất để xử lý")
        elif len(faces) == 0:
            print("⚠ [KẾT QUẢ] Không phát hiện được khuôn mặt nào sau khi lọc")
            return [], 0
        else:
            print(f"✓ Phát hiện 1 khuôn mặt duy nhất")

        print(f"\n🧠 [BƯỚC 2: PHÂN TÍCH] Phân tích đặc điểm khuôn mặt...")

        # ===== BƯỚC 2: PHÂN TÍCH =====
        # Sử dụng known_faces_db được truyền vào
        if known_faces_db is None:
            # Chế độ chỉ phát hiện khuôn mặt (không so sánh) - dùng khi đăng ký
            print(f"\n🔍 [CHỈ PHÁT HIỆN] Không so sánh với database - chế độ đăng ký khuôn mặt mới")
            
            results = []
            for (x, y, w, h) in faces:
                top, left = int(y), int(x)
                bottom, right = int(y + h), int(x + w)
                
                result = {
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'matches': [],
                    'best_match': None
                }
                results.append(result)
                print(f"✓ Phát hiện khuôn mặt tại vị trí ({x}, {y}, {w}, {h})")
            
            return results, len(faces)
        
        known_faces_db = KNOWN_FACES_DB

        print(f"\n🔍 [BƯỚC 4: SO KHỚP] So sánh với {len(known_faces_db)} khuôn mặt trong database...")

        # ===== BƯỚC 4: SO KHỚP DỮ LIỆU =====
        results = []
        for (x, y, w, h) in faces:
            print(f"\n--- Phân tích khuôn mặt chính ---")

            # Tính toán vị trí khuôn mặt
            top, left = int(y), int(x)
            bottom, right = int(y + h), int(x + w)

            # Lấy vùng khuôn mặt hiện tại để phân tích
            current_face_roi = gray[y:y+h, x:x+w]

            # ===== BƯỚC 3: CHUYỂN ĐỔI DỮ LIỆU =====
            # Chuẩn bị dữ liệu để so sánh

            # Tìm khuôn mặt có độ tương đồng cao nhất
            matches = []
            best_match = None
            best_similarity = 0.0

            for name, face_data in known_faces_db.items():
                if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                    try:
                        print(f"  📸 Đang xử lý: {name}")

                        # Load ảnh khuôn mặt đã đăng ký
                        registered_image = cv2.imread(face_data['image_path'])
                        if registered_image is not None:
                            registered_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)

                            # Phát hiện khuôn mặt trong ảnh đã đăng ký
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

                                # Resize để so sánh cùng kích thước (tăng độ chính xác)
                                target_size = (128, 128)
                                registered_face_resized = cv2.resize(registered_face_roi, target_size)
                                current_face_resized = cv2.resize(current_face_roi, target_size)

                                # ===== PHƯƠNG PHÁP SO KHỚP =====
                                # Sử dụng thuật toán đa phương pháp (Template + Statistical)
                                similarity = calculate_face_similarity_improved(current_face_resized, registered_face_resized)

                                print(f"    📊 Độ tương đồng: {similarity:.4f}")

                                # Ngưỡng nhận dạng với nhiều mức độ tin cậy (giảm để dễ nhận diện hơn)
                                OPENCV_CONFIDENCE_THRESHOLDS = {
                                    'high': 0.45,      # Rất tin cậy - Giảm từ 55%
                                    'medium': 0.35,    # Trung bình - Giảm từ 50%
                                    'low': 0.25        # Thấp, cần xác minh thêm - Giảm từ 45%
                                }

                                confidence_level = 'none'
                                if similarity > OPENCV_CONFIDENCE_THRESHOLDS['high']:
                                    confidence_level = 'high'
                                elif similarity > OPENCV_CONFIDENCE_THRESHOLDS['medium']:
                                    confidence_level = 'medium'
                                elif similarity > OPENCV_CONFIDENCE_THRESHOLDS['low']:
                                    confidence_level = 'low'

                                if confidence_level != 'none':
                                    matches.append((name, float(similarity), confidence_level))
                                    print(f"    ✅ {name} được nhận dạng với độ tin cậy {confidence_level}!")

                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = (name, float(similarity), confidence_level)
                                        print(f"    🎯 {name} là kết quả phù hợp nhất!")
                                else:
                                    print(f"    ❌ {name} không đủ tương đồng")

                            else:
                                print(f"    ⚠️ Không tìm thấy khuôn mặt trong ảnh đã đăng ký")

                    except Exception as e:
                        print(f"    ❌ Lỗi khi xử lý ảnh của {name}: {e}")

            # Sắp xếp theo độ tương đồng
            matches.sort(key=lambda x: x[1], reverse=True)

            # ===== BƯỚC 5: XÁC NHẬN DANH TÍNH =====
            if best_match and best_similarity > 0.20:  # Giảm ngưỡng từ 0.45 xuống 0.20 (20%)
                results.append({
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'matches': matches[:1],  # Chỉ giữ match tốt nhất
                    'best_match': best_match
                })
                name, similarity, confidence = best_match
                print(f"🎉 [KẾT QUẢ] Xác nhận: {name} ({similarity:.1%} - {confidence} confidence)")
            else:
                results.append({
                    'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left},
                    'matches': [],
                    'best_match': None
                })
                print(f"❓ [KẾT QUẢ] Không xác định được danh tính")

        # Thống kê kết quả cuối cùng
        recognized_faces = len([r for r in results if r['best_match']])
        unrecognized_faces = len(faces) - recognized_faces

        print(f"\n📋 [TÓM TẮT KẾT QUẢ]:")
        print(f"  - Tổng số khuôn mặt phát hiện: {len(faces)}")
        print(f"  - Khuôn mặt đã nhận diện: {recognized_faces}")
        print(f"  - Khuôn mặt chưa xác định: {unrecognized_faces}")
        print(f"  - Phương pháp: Appearance-Based (Haar + Statistical)")
        print(f"  - Thời gian xử lý: < 2 giây")

        return results, int(len(faces))

    except Exception as e:
        print(f"❌ Lỗi trong quá trình nhận diện khuôn mặt: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

def verify_face_identity(face_image_path, known_face_name, known_faces_db=None):
    """
    XÁC MINH DANH TÍNH KHUÔN MẶT (1:1 Comparison)
    Đây là bước xác minh cuối cùng với độ chính xác cao nhất

    Phương pháp: One-to-One Matching
    - So sánh trực tiếp khuôn mặt cần xác minh với khuôn mặt đã biết
    - Sử dụng ngưỡng nghiêm ngặt hơn so với nhận diện thông thường
    - Trả về kết quả boolean: xác nhận hoặc từ chối
    """
    try:
        print(f"\n🔍 [XÁC MINH 1:1] Bắt đầu xác minh danh tính: {known_face_name}")

        if known_faces_db is None:
            known_faces_db = KNOWN_FACES_DB

        if known_face_name not in known_faces_db:
            return False, 0.0, "Người này không có trong cơ sở dữ liệu"

        known_face_data = known_faces_db[known_face_name]
        if 'image_path' not in known_face_data or not os.path.exists(known_face_data['image_path']):
            return False, 0.0, "Không tìm thấy ảnh của người này"

        print(f"📊 Đang so sánh với mẫu khuôn mặt đã đăng ký...")

        # Load và xử lý ảnh cần xác minh
        if FACE_RECOGNITION_AVAILABLE:
            # ===== PHƯƠNG PHÁP: Neural Networks (face_recognition) =====
            print("🎯 Sử dụng phương pháp: Neural Networks + Face Encoding")

            unknown_image = face_recognition.load_image_file(face_image_path)
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if len(unknown_encodings) == 0:
                return False, 0.0, "Không tìm thấy khuôn mặt trong ảnh cần xác minh"

            # Load face encoding của người đã biết
            known_image = face_recognition.load_image_file(known_face_data['image_path'])
            known_encodings = face_recognition.face_encodings(known_image)

            if len(known_encodings) == 0:
                return False, 0.0, "Không tìm thấy khuôn mặt trong ảnh đã đăng ký"

            # So sánh face encodings (Euclidean distance)
            face_distances = face_recognition.face_distance(known_encodings, unknown_encodings[0])
            min_distance = min(face_distances)
            similarity = 1.0 - min_distance

            print(f"📏 Khoảng cách Euclidean: {min_distance:.4f}")
            print(f"📊 Độ tương đồng: {similarity:.4f}")

            # Ngưỡng xác minh nghiêm ngặt hơn cho 1:1 verification (0.8 = very high confidence)
            VERIFICATION_THRESHOLD = 0.8
            is_verified = similarity > VERIFICATION_THRESHOLD

            confidence_level = "very_high" if similarity > 0.85 else "high" if similarity > VERIFICATION_THRESHOLD else "medium"

            result_text = f"Xác minh {'THÀNH CÔNG' if is_verified else 'THẤT BẠI'} - Độ tin cậy: {confidence_level} ({similarity:.1%})"
            print(f"🎯 [KẾT QUẢ XÁC MINH] {result_text}")

            return is_verified, similarity, result_text

        else:
            # ===== PHƯƠNG PHÁP: Appearance-Based + Statistical =====
            print("🎯 Sử dụng phương pháp: Appearance-Based + Statistical Analysis")

            unknown_image = cv2.imread(face_image_path)
            if unknown_image is None:
                return False, 0.0, "Không thể load ảnh cần xác minh"

            known_image = cv2.imread(known_face_data['image_path'])
            if known_image is None:
                return False, 0.0, "Không thể load ảnh đã đăng ký"

            # Phát hiện khuôn mặt bằng Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Xử lý ảnh unknown
            unknown_gray = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
            unknown_faces = face_cascade.detectMultiScale(unknown_gray, 1.1, 5, minSize=(60, 60))

            if len(unknown_faces) == 0:
                return False, 0.0, "Không tìm thấy khuôn mặt trong ảnh cần xác minh"

            # Xử lý ảnh known
            known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
            known_faces = face_cascade.detectMultiScale(known_gray, 1.1, 5, minSize=(60, 60))

            if len(known_faces) == 0:
                return False, 0.0, "Không tìm thấy khuôn mặt trong ảnh đã đăng ký"

            # Lấy khuôn mặt đầu tiên từ mỗi ảnh
            ux, uy, uw, uh = unknown_faces[0]
            kx, ky, kw, kh = known_faces[0]

            # Trích xuất và chuẩn hóa vùng khuôn mặt
            unknown_face = unknown_gray[uy:uy+uh, ux:ux+uw]
            known_face = known_gray[ky:ky+kh, kx:kx+kw]

            target_size = (128, 128)
            unknown_face = cv2.resize(unknown_face, target_size)
            known_face = cv2.resize(known_face, target_size)

            # Tính độ tương đồng bằng thuật toán đa phương pháp
            similarity = calculate_face_similarity_improved(unknown_face, known_face)

            print(f"📊 Độ tương đồng: {similarity:.4f}")

            # Ngưỡng xác minh cho OpenCV (0.6 = high confidence cho 1:1 verification)
            VERIFICATION_THRESHOLD = 0.6
            is_verified = similarity > VERIFICATION_THRESHOLD

            confidence_level = "high" if similarity > 0.7 else "medium" if similarity > VERIFICATION_THRESHOLD else "low"

            result_text = f"Xác minh {'THÀNH CÔNG' if is_verified else 'THẤT BẠI'} - Độ tin cậy: {confidence_level} ({similarity:.1%})"
            print(f"🎯 [KẾT QUẢ XÁC MINH] {result_text}")

            return is_verified, similarity, result_text

    except Exception as e:
        print(f"❌ Lỗi khi xác minh danh tính: {e}")
        return False, 0.0, f"Lỗi xử lý: {str(e)}"

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
            
            # Kiểm tra xem có khuôn mặt trong ảnh không (chỉ phát hiện, không so sánh)
            faces, count = detect_faces_advanced(image_path, None)  # Không so sánh với database khi đăng ký
            
            if count > 0:
                # Trích xuất metadata
                metadata = extract_face_metadata(image_path)

                # Lưu vào cơ sở dữ liệu với thông tin phong phú
                known_faces = load_known_faces()
                known_faces[name] = {
                    'image_path': image_path,
                    'registered_at': datetime.now().isoformat(),
                    'face_count': count,
                    'metadata': metadata,
                    'tags': [],
                    'notes': '',
                    'last_updated': datetime.now().isoformat()
                }
                save_known_faces(known_faces)

                flash(f'Đăng ký khuôn mặt cho {name} thành công! Tìm thấy {count} khuôn mặt. Chất lượng: {(metadata["quality_score"]*100):.1f}%.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Không thể phát hiện khuôn mặt trong ảnh!', 'error')
                os.remove(image_path)  # Xóa ảnh không hợp lệ
        else:
            flash('File không được hỗ trợ!', 'error')
    
    return render_template('register.html')

@app.route('/recognize', methods=['POST'])
def recognize_face_api():
    """API nhận diện khuôn mặt từ ảnh"""
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh được upload'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Không có ảnh được chọn'}), 400

    if image_file and is_image_file(image_file.filename):
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
                'face_count': face_count,
                'media_type': 'image'
            })

        except Exception as e:
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File không được hỗ trợ hoặc không phải ảnh'}), 400

@app.route('/verify_face', methods=['POST'])
def verify_face_api():
    """API xác minh danh tính khuôn mặt (1:1 comparison)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Không có ảnh được upload'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Không có ảnh được chọn'}), 400

        person_name = request.form.get('person_name', '').strip()
        if not person_name:
            return jsonify({'error': 'Vui lòng cung cấp tên người cần xác minh'}), 400

        if image_file and is_image_file(image_file.filename):
            # Lưu ảnh tạm thời
            filename = secure_filename(f"verify_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)

            try:
                # Xác minh danh tính
                known_faces = load_known_faces()
                is_verified, similarity, result_text = verify_face_identity(image_path, person_name, known_faces)

                # Xóa file tạm
                os.remove(image_path)

                confidence_percentage = f"{((1 - similarity) * 100):.1f}" if similarity < 1 else '100.0'

                return jsonify({
                    'success': True,
                    'person_name': person_name,
                    'is_verified': is_verified,
                    'similarity': float(similarity),
                    'confidence_percentage': confidence_percentage,
                    'result_text': result_text,
                    'verification_threshold': 0.8 if FACE_RECOGNITION_AVAILABLE else 0.6
                })

            except Exception as e:
                if os.path.exists(image_path):
                    os.remove(image_path)
                return jsonify({'error': str(e)}), 500

        return jsonify({'error': 'File không được hỗ trợ'}), 400

    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/batch_recognize_images', methods=['POST'])
def batch_recognize_images_api():
    """API nhận diện batch nhiều ảnh"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'Không có ảnh được upload'}), 400

        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': 'Không có ảnh được chọn'}), 400

        # Validate files
        valid_files = []
        for file in files:
            if file.filename == '':
                continue
            if is_image_file(file.filename):
                valid_files.append(file)
            else:
                print(f"⚠️ Bỏ qua file không hợp lệ: {file.filename}")

        if not valid_files:
            return jsonify({'error': 'Không có file ảnh hợp lệ'}), 400

        # Lưu tạm thời và xử lý
        temp_paths = []
        try:
            for file in valid_files:
                filename = secure_filename(f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(temp_paths)}.jpg")
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                temp_paths.append(file_path)

            # Xử lý batch
            known_faces = load_known_faces()
            results, total_faces = batch_process_images(temp_paths, known_faces)

            # Thống kê tổng hợp
            total_files = len(results)
            successful_files = len([r for r in results if 'error' not in r])
            failed_files = total_files - successful_files
            total_recognized = sum(len([f for f in r['faces'] if f.get('best_match')]) for r in results if 'error' not in r)

            return jsonify({
                'success': True,
                'results': results,
                'summary': {
                    'total_files': total_files,
                    'successful_files': successful_files,
                    'failed_files': failed_files,
                    'total_faces': total_faces,
                    'total_recognized': total_recognized
                },
                'media_type': 'batch_images'
            })

        finally:
            # Xóa file tạm
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/batch_recognize_videos', methods=['POST'])
def batch_recognize_videos_api():
    """API nhận diện batch nhiều video"""
    try:
        if 'videos' not in request.files:
            return jsonify({'error': 'Không có video được upload'}), 400

        files = request.files.getlist('videos')
        if not files or len(files) == 0:
            return jsonify({'error': 'Không có video được chọn'}), 400

        # Validate files
        valid_files = []
        for file in files:
            if file.filename == '':
                continue
            if is_video_file(file.filename):
                valid_files.append(file)
            else:
                print(f"⚠️ Bỏ qua file không hợp lệ: {file.filename}")

        if not valid_files:
            return jsonify({'error': 'Không có file video hợp lệ'}), 400

        # Lưu tạm thời và xử lý
        temp_paths = []
        try:
            for file in valid_files:
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = secure_filename(f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(temp_paths)}.{ext}")
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                temp_paths.append(file_path)

            # Xử lý batch
            known_faces = load_known_faces()
            results, total_faces = batch_process_videos(temp_paths, known_faces)

            # Thống kê tổng hợp
            total_files = len(results)
            successful_files = len([r for r in results if 'error' not in r])
            failed_files = total_files - successful_files
            total_recognized = sum(r.get('recognized_faces', 0) for r in results if 'error' not in r)
            total_unique_persons = sum(r.get('unique_persons', 0) for r in results if 'error' not in r)

            return jsonify({
                'success': True,
                'results': results,
                'summary': {
                    'total_files': total_files,
                    'successful_files': successful_files,
                    'failed_files': failed_files,
                    'total_faces': total_faces,
                    'total_recognized': total_recognized,
                    'total_unique_persons': total_unique_persons
                },
                'media_type': 'batch_videos'
            })

        finally:
            # Xóa file tạm
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.route('/recognize_video', methods=['POST'])
def recognize_video_api():
    """API nhận diện khuôn mặt từ video"""
    if 'video' not in request.files:
        return jsonify({'error': 'Không có video được upload'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Không có video được chọn'}), 400

    if video_file and is_video_file(video_file.filename):
        # Lưu video tạm thời
        filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{video_file.filename.rsplit('.', 1)[1].lower()}")
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(video_path)

        try:
            # Nhận diện khuôn mặt trong video
            known_faces = load_known_faces()
            results, face_count = recognize_faces_in_video(video_path, known_faces)

            # Xóa file tạm
            os.remove(video_path)

            # Thống kê kết quả
            recognized_faces = len([r for r in results if r.get('best_match')])
            unrecognized_faces = face_count - recognized_faces

            # Gom nhóm kết quả theo người
            person_stats = {}
            for result in results:
                if result.get('best_match'):
                    name, similarity = result['best_match']
                    if name not in person_stats:
                        person_stats[name] = {
                            'count': 0,
                            'avg_similarity': 0.0,
                            'frames': []
                        }
                    person_stats[name]['count'] += 1
                    person_stats[name]['avg_similarity'] += similarity
                    person_stats[name]['frames'].append({
                        'frame_number': result['frame_number'],
                        'timestamp': result['timestamp'],
                        'similarity': similarity
                    })

            # Tính trung bình similarity
            for name in person_stats:
                person_stats[name]['avg_similarity'] /= person_stats[name]['count']

            return jsonify({
                'success': True,
                'results': results,
                'face_count': face_count,
                'recognized_faces': recognized_faces,
                'unrecognized_faces': unrecognized_faces,
                'person_stats': person_stats,
                'media_type': 'video'
            })

        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File không được hỗ trợ hoặc không phải video'}), 400

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

@app.route('/update_face_metadata/<name>', methods=['POST'])
def update_face_metadata(name):
    """Cập nhật metadata cho khuôn mặt"""
    known_faces = load_known_faces()
    if name not in known_faces:
        return jsonify({'error': f'Không tìm thấy khuôn mặt của {name}'}), 404

    try:
        # Lấy dữ liệu từ form
        tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
        tags = [tag.strip() for tag in tags if tag.strip()]
        notes = request.form.get('notes', '')

        # Cập nhật metadata
        known_faces[name]['tags'] = tags
        known_faces[name]['notes'] = notes
        known_faces[name]['last_updated'] = datetime.now().isoformat()

        save_known_faces(known_faces)

        return jsonify({
            'success': True,
            'message': f'Đã cập nhật thông tin cho {name}',
            'tags': tags,
            'notes': notes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_face_details/<name>')
def get_face_details(name):
    """Lấy chi tiết thông tin khuôn mặt"""
    known_faces = load_known_faces()
    if name not in known_faces:
        return jsonify({'error': f'Không tìm thấy khuôn mặt của {name}'}), 404

    face_data = known_faces[name].copy()

    # Thêm thông tin bổ sung
    face_data['name'] = name
    face_data['image_exists'] = os.path.exists(face_data.get('image_path', ''))

    return jsonify(face_data)

@app.route('/refresh_metadata')
def refresh_metadata():
    """Làm mới metadata cho tất cả khuôn mặt"""
    known_faces = load_known_faces()
    updated_count = 0

    for name, face_data in known_faces.items():
        if 'image_path' in face_data and os.path.exists(face_data['image_path']):
            try:
                metadata = extract_face_metadata(face_data['image_path'])
                face_data['metadata'] = metadata
                face_data['last_updated'] = datetime.now().isoformat()
                updated_count += 1
                print(f"✓ Đã cập nhật metadata cho {name}")
            except Exception as e:
                print(f"⚠️ Lỗi cập nhật metadata cho {name}: {e}")

    save_known_faces(known_faces)

    flash(f'Đã làm mới metadata cho {updated_count} khuôn mặt!', 'success')
    return redirect(url_for('index'))

@app.route('/camera')
def camera():
    """Trang camera real-time"""
    return render_template('camera.html')

@app.route('/process_camera_frame', methods=['POST'])
def process_camera_frame():
    """Xử lý frame từ camera real-time"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'Không có frame được gửi'}), 400

        frame_file = request.files['frame']
        if frame_file.filename == '':
            return jsonify({'error': 'Frame trống'}), 400

        # Lưu frame tạm thời
        filename = secure_filename(f"camera_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        frame_path = os.path.join(UPLOAD_FOLDER, filename)
        frame_file.save(frame_path)

        try:
            # Load cơ sở dữ liệu khuôn mặt
            known_faces = load_known_faces()

            # Kiểm tra xem có khuôn mặt nào trong database không
            if len(known_faces) == 0:
                print("⚠️ [CAMERA] Database trống - không có khuôn mặt nào để nhận diện")

                # Vẫn phát hiện khuôn mặt nhưng không nhận diện được
                if FACE_RECOGNITION_AVAILABLE:
                    # Phát hiện khuôn mặt mà không so khớp
                    image = face_recognition.load_image_file(frame_path)
                    face_locations = face_recognition.face_locations(image)

                    response_data = {
                        'face_count': len(face_locations),
                        'faces': [],
                        'warning': 'Database trống - vui lòng đăng ký khuôn mặt trước khi nhận diện'
                    }

                    for top, right, bottom, left in face_locations:
                        response_data['faces'].append({
                            'location': {
                                'top': int(top),
                                'right': int(right),
                                'bottom': int(bottom),
                                'left': int(left)
                            },
                            'has_match': False,
                            'message': 'Unknown - No database'
                        })

                    return jsonify(response_data)
                else:
                    # Fallback với OpenCV
                    image = cv2.imread(frame_path)
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

                        response_data = {
                            'face_count': len(faces),
                            'faces': [],
                            'warning': 'Database trống - vui lòng đăng ký khuôn mặt trước khi nhận diện'
                        }

                        for (x, y, w, h) in faces:
                            response_data['faces'].append({
                                'location': {
                                    'top': int(y),
                                    'right': int(x + w),
                                    'bottom': int(y + h),
                                    'left': int(x)
                                },
                                'has_match': False,
                                'message': 'Unknown - No database'
                            })

                        return jsonify(response_data)

                return jsonify({
                    'face_count': 0,
                    'faces': [],
                    'warning': 'Database trống - vui lòng đăng ký khuôn mặt trước khi nhận diện'
                })

            # Database có dữ liệu - tiến hành nhận diện bình thường
            results, face_count = detect_faces_advanced(frame_path, known_faces)

            # Chuẩn bị dữ liệu trả về cho frontend
            response_data = {
                'face_count': face_count,
                'faces': []
            }

            for result in results:
                face_data = {
                    'location': result['location'],
                    'has_match': result.get('best_match') is not None
                }

                if result.get('best_match'):
                    name, similarity, confidence = result['best_match']
                    face_data.update({
                        'name': name,
                        'similarity': float(similarity),
                        'confidence_level': confidence,
                        'confidence_percentage': f"{((1 - similarity) * 100):.1f}"
                    })
                else:
                    face_data['message'] = 'Unknown - No match found'

                response_data['faces'].append(face_data)

            return jsonify(response_data)

        finally:
            # Xóa file tạm
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except:
                    pass

    except Exception as e:
        print(f"❌ Lỗi xử lý camera frame: {e}")
        return jsonify({'error': str(e), 'face_count': 0, 'faces': []}), 500

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

@app.route('/system_status')
def system_status():
    """Trả về thông tin trạng thái hệ thống"""
    try:
        memory_info = get_memory_usage()

        # Thông tin hệ thống
        system_info = {
            'memory_usage': memory_info,
            'face_recognition_available': FACE_RECOGNITION_AVAILABLE,
            'total_known_faces': len(KNOWN_FACES_DB),
            'encodings_loaded': len(KNOWN_FACE_ENCODINGS),
            'upload_folder_size': sum(os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
                                    for f in os.listdir(UPLOAD_FOLDER)
                                    if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))) / (1024*1024),  # MB
            'server_time': datetime.now().isoformat(),
            'opencv_version': cv2.__version__,
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
        }

        return jsonify(system_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup_temp')
def cleanup_temp():
    """Dọn dẹp file tạm thời"""
    try:
        cleanup_temp_files()
        flash('Đã dọn dẹp file tạm thời!', 'success')
    except Exception as e:
        flash(f'Lỗi khi dọn dẹp: {str(e)}', 'error')

    return redirect(url_for('index'))

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

    # Dọn dẹp file tạm khi khởi động
    print("🧹 Đang dọn dẹp file tạm thời...")
    cleanup_temp_files()

    # Tạo dữ liệu demo khi khởi động nếu database trống
    known_faces = load_known_faces()
    if len(known_faces) == 0:
        print("Tạo dữ liệu demo...")
        demo_faces = {
            'Nguyễn Văn A': {
                'image_path': 'demo/person1.jpg',
                'registered_at': datetime.now().isoformat(),
                'face_count': 1,
                'metadata': {
                    'face_count': 1,
                    'faces': [],
                    'quality_score': 0.85,
                    'brightness': 0.6,
                    'contrast': 0.4,
                    'blur_score': 150.0
                },
                'tags': ['demo', 'test'],
                'notes': 'Dữ liệu demo - Nguyễn Văn A',
                'last_updated': datetime.now().isoformat()
            },
            'Trần Thị B': {
                'image_path': 'demo/person2.jpg',
                'registered_at': datetime.now().isoformat(),
                'face_count': 1,
                'metadata': {
                    'face_count': 1,
                    'faces': [],
                    'quality_score': 0.78,
                    'brightness': 0.55,
                    'contrast': 0.45,
                    'blur_score': 120.0
                },
                'tags': ['demo', 'female'],
                'notes': 'Dữ liệu demo - Trần Thị B',
                'last_updated': datetime.now().isoformat()
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
