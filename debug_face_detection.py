#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ debug phát hiện khuôn mặt
Debug tool for face detection issues
"""

import os
import cv2
import numpy as np
from datetime import datetime

def debug_face_detection(image_path):
    """Debug chi tiết quá trình phát hiện khuôn mặt"""
    print(f"\n🔍 [DEBUG] Bắt đầu debug phát hiện khuôn mặt: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return
    
    # Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể load ảnh")
        return
    
    print(f"📊 Thông tin ảnh:")
    print(f"  - Kích thước: {image.shape}")
    print(f"  - Loại: {image.dtype}")
    print(f"  - Kênh màu: {image.shape[2] if len(image.shape) > 2 else 1}")
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"  - Grayscale shape: {gray.shape}")
    
    # Kiểm tra độ sáng trung bình
    brightness = np.mean(gray)
    contrast = np.std(gray)
    print(f"  - Độ sáng trung bình: {brightness:.2f}/255")
    print(f"  - Độ tương phản: {contrast:.2f}")
    
    # Test với các tham số khác nhau
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Không thể load Haar Cascade classifier")
        return
    
    print(f"\n🧪 [TEST] Thử các tham số phát hiện khuôn mặt khác nhau:")
    
    # Test 1: Tham số mặc định (nghiêm ngặt)
    faces1 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50),
        maxSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"  - Tham số nghiêm ngặt: {len(faces1)} khuôn mặt")
    
    # Test 2: Tham số trung bình
    faces2 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"  - Tham số trung bình: {len(faces2)} khuôn mặt")
    
    # Test 3: Tham số lỏng lẻo
    faces3 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=2,
        minSize=(20, 20),
        maxSize=(500, 500),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"  - Tham số lỏng lẻo: {len(faces3)} khuôn mặt")
    
    # Test 4: Tham số rất lỏng lẻo
    faces4 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=1,
        minSize=(10, 10),
        maxSize=(1000, 1000),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(f"  - Tham số rất lỏng lẻo: {len(faces4)} khuôn mặt")
    
    # Chọn kết quả tốt nhất
    best_faces = faces4 if len(faces4) > 0 else faces3 if len(faces3) > 0 else faces2 if len(faces2) > 0 else faces1
    print(f"\n✅ Kết quả tốt nhất: {len(best_faces)} khuôn mặt")
    
    if len(best_faces) > 0:
        print(f"\n📋 Chi tiết các khuôn mặt phát hiện được:")
        for i, (x, y, w, h) in enumerate(best_faces):
            print(f"  - Khuôn mặt {i+1}: vị trí=({x}, {y}), kích thước=({w}, {h})")
            
            # Kiểm tra chất lượng khuôn mặt
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_brightness = np.mean(face_roi)
                face_contrast = np.std(face_roi)
                aspect_ratio = w / h
                
                print(f"    + Độ sáng: {face_brightness:.2f}/255")
                print(f"    + Độ tương phản: {face_contrast:.2f}")
                print(f"    + Tỷ lệ khung hình: {aspect_ratio:.2f}")
                
                # Đánh giá chất lượng
                quality_score = 0
                if 30 <= face_brightness <= 200:
                    quality_score += 1
                if face_contrast >= 20:
                    quality_score += 1
                if 0.7 <= aspect_ratio <= 1.3:
                    quality_score += 1
                
                print(f"    + Điểm chất lượng: {quality_score}/3")
    else:
        print(f"\n❌ Không phát hiện được khuôn mặt nào!")
        print(f"\n💡 Gợi ý khắc phục:")
        print(f"  - Kiểm tra ảnh có rõ nét không")
        print(f"  - Đảm bảo khuôn mặt chiếm ít nhất 10% diện tích ảnh")
        print(f"  - Thử ảnh có ánh sáng tốt hơn")
        print(f"  - Kiểm tra khuôn mặt có bị che khuất không")
        print(f"  - Thử ảnh có góc nhìn thẳng mặt")

def test_with_different_images():
    """Test với các ảnh trong thư mục uploads"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"❌ Thư mục {uploads_dir} không tồn tại")
        return
    
    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong {uploads_dir}")
        return
    
    print(f"📁 Tìm thấy {len(image_files)} ảnh trong {uploads_dir}")
    
    for image_file in image_files[:5]:  # Test tối đa 5 ảnh
        image_path = os.path.join(uploads_dir, image_file)
        print(f"\n{'='*60}")
        debug_face_detection(image_path)

if __name__ == "__main__":
    print("🔧 CÔNG CỤ DEBUG PHÁT HIỆN KHUÔN MẶT")
    print("="*50)
    
    # Test với ảnh cụ thể nếu có
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        debug_face_detection(image_path)
    else:
        # Test với tất cả ảnh trong uploads
        test_with_different_images()
