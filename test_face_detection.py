#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test phát hiện khuôn mặt với ảnh mẫu
Test face detection with sample images
"""

import os
import cv2
import numpy as np
from simple_web_app import detect_faces_advanced, detect_faces_simple

def create_test_image():
    """Tạo ảnh test đơn giản với hình chữ nhật giả khuôn mặt"""
    # Tạo ảnh trắng 400x400
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Vẽ hình chữ nhật giả khuôn mặt
    cv2.rectangle(img, (150, 100), (250, 200), (0, 0, 0), 2)
    
    # Vẽ mắt
    cv2.circle(img, (170, 130), 5, (0, 0, 0), -1)
    cv2.circle(img, (230, 130), 5, (0, 0, 0), -1)
    
    # Vẽ mũi
    cv2.circle(img, (200, 150), 3, (0, 0, 0), -1)
    
    # Vẽ miệng
    cv2.ellipse(img, (200, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Lưu ảnh test
    test_path = "test_face.jpg"
    cv2.imwrite(test_path, img)
    print(f"✅ Đã tạo ảnh test: {test_path}")
    return test_path

def test_face_detection():
    """Test phát hiện khuôn mặt"""
    print("🧪 [TEST] Bắt đầu test phát hiện khuôn mặt")
    
    # Tạo ảnh test
    test_image = create_test_image()
    
    if not os.path.exists(test_image):
        print("❌ Không thể tạo ảnh test")
        return
    
    print(f"\n📸 Test với ảnh: {test_image}")
    
    # Test với detect_faces_simple (OpenCV)
    print(f"\n🔍 [TEST 1] Sử dụng OpenCV (detect_faces_simple):")
    try:
        results, count = detect_faces_simple(test_image, None)
        print(f"Kết quả: {count} khuôn mặt phát hiện được")
        if count > 0:
            for i, result in enumerate(results):
                print(f"  - Khuôn mặt {i+1}: {result['location']}")
        else:
            print("  ❌ Không phát hiện được khuôn mặt nào")
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
    
    # Test với detect_faces_advanced (face_recognition)
    print(f"\n🔍 [TEST 2] Sử dụng face_recognition (detect_faces_advanced):")
    try:
        results, count = detect_faces_advanced(test_image, None)
        print(f"Kết quả: {count} khuôn mặt phát hiện được")
        if count > 0:
            for i, result in enumerate(results):
                print(f"  - Khuôn mặt {i+1}: {result['location']}")
        else:
            print("  ❌ Không phát hiện được khuôn mặt nào")
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
    
    # Dọn dẹp
    if os.path.exists(test_image):
        os.remove(test_image)
        print(f"\n🗑️ Đã xóa ảnh test: {test_image}")

if __name__ == "__main__":
    test_face_detection()
