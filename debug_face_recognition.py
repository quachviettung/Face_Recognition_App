#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ debug nhận diện khuôn mặt
Debug tool for face recognition issues
"""

import os
import json
from simple_web_app import load_known_faces, KNOWN_FACE_ENCODINGS, FACE_RECOGNITION_AVAILABLE

def debug_face_database():
    """Debug cơ sở dữ liệu khuôn mặt"""
    print("🔍 [DEBUG] Kiểm tra cơ sở dữ liệu khuôn mặt")
    print("="*50)
    
    # Load database
    known_faces = load_known_faces()
    print(f"📊 Số lượng khuôn mặt trong database: {len(known_faces)}")
    
    if len(known_faces) == 0:
        print("❌ Database trống! Vui lòng đăng ký khuôn mặt trước.")
        return
    
    print(f"\n📋 Danh sách khuôn mặt đã đăng ký:")
    for name, data in known_faces.items():
        print(f"  - {name}:")
        print(f"    + Ảnh: {data.get('image_path', 'N/A')}")
        print(f"    + Tồn tại: {'✅' if os.path.exists(data.get('image_path', '')) else '❌'}")
        print(f"    + Số khuôn mặt: {data.get('face_count', 'N/A')}")
        print(f"    + Đăng ký lúc: {data.get('registered_at', 'N/A')}")
    
    # Kiểm tra face encodings
    print(f"\n🔐 Face encodings:")
    print(f"  - Số lượng encodings: {len(KNOWN_FACE_ENCODINGS)}")
    print(f"  - Face recognition available: {'✅' if FACE_RECOGNITION_AVAILABLE else '❌'}")
    
    if FACE_RECOGNITION_AVAILABLE:
        for name, encoding in KNOWN_FACE_ENCODINGS.items():
            print(f"  - {name}: encoding shape = {encoding.shape}")
    else:
        print("  ⚠️ Sử dụng OpenCV fallback - không có face encodings")

def debug_face_comparison(image1_path, image2_path):
    """Debug so sánh 2 ảnh khuôn mặt"""
    print(f"\n🔍 [DEBUG] So sánh 2 ảnh khuôn mặt")
    print("="*50)
    
    if not os.path.exists(image1_path):
        print(f"❌ Ảnh 1 không tồn tại: {image1_path}")
        return
    
    if not os.path.exists(image2_path):
        print(f"❌ Ảnh 2 không tồn tại: {image2_path}")
        return
    
    print(f"📸 Ảnh 1: {image1_path}")
    print(f"📸 Ảnh 2: {image2_path}")
    
    if FACE_RECOGNITION_AVAILABLE:
        try:
            import face_recognition
            
            # Load và encode ảnh 1
            img1 = face_recognition.load_image_file(image1_path)
            encodings1 = face_recognition.face_encodings(img1)
            
            if len(encodings1) == 0:
                print("❌ Không tìm thấy khuôn mặt trong ảnh 1")
                return
            
            # Load và encode ảnh 2
            img2 = face_recognition.load_image_file(image2_path)
            encodings2 = face_recognition.face_encodings(img2)
            
            if len(encodings2) == 0:
                print("❌ Không tìm thấy khuôn mặt trong ảnh 2")
                return
            
            # So sánh
            distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
            similarity = 1.0 - distance
            
            print(f"\n📊 Kết quả so sánh:")
            print(f"  - Khoảng cách Euclidean: {distance:.4f}")
            print(f"  - Độ tương đồng: {similarity:.4f} ({similarity:.1%})")
            
            # Đánh giá
            if similarity > 0.6:
                print(f"  - Kết luận: ✅ CÓ THỂ là cùng một người (High confidence)")
            elif similarity > 0.5:
                print(f"  - Kết luận: ⚠️ CÓ THỂ là cùng một người (Medium confidence)")
            elif similarity > 0.4:
                print(f"  - Kết luận: ❓ CÓ THỂ là cùng một người (Low confidence)")
            else:
                print(f"  - Kết luận: ❌ KHÔNG PHẢI cùng một người")
                
        except Exception as e:
            print(f"❌ Lỗi khi so sánh: {e}")
    else:
        print("❌ Face recognition không khả dụng, không thể so sánh chi tiết")

def suggest_improvements():
    """Đưa ra gợi ý cải thiện"""
    print(f"\n💡 [GỢI Ý] Cách cải thiện nhận diện khuôn mặt:")
    print("="*50)
    
    print("1. 📸 Chất lượng ảnh đăng ký:")
    print("   - Ảnh rõ nét, không bị mờ")
    print("   - Ánh sáng đủ, không quá tối hoặc quá sáng")
    print("   - Khuôn mặt chiếm ít nhất 20% diện tích ảnh")
    print("   - Góc nhìn thẳng mặt, không nghiêng quá nhiều")
    print("   - Không đeo kính râm, mũ che mặt")
    
    print("\n2. 📹 Điều kiện camera:")
    print("   - Đảm bảo ánh sáng tốt khi sử dụng camera")
    print("   - Giữ khoảng cách phù hợp (không quá gần/xa)")
    print("   - Nhìn thẳng vào camera")
    print("   - Tránh chuyển động quá nhanh")
    
    print("\n3. 🔧 Cài đặt hệ thống:")
    print("   - Ngưỡng nhận diện đã được giảm để dễ nhận diện hơn")
    print("   - Thử đăng ký nhiều ảnh khác nhau của cùng một người")
    print("   - Kiểm tra ảnh đăng ký có tồn tại không")
    
    print("\n4. 🐛 Debug:")
    print("   - Kiểm tra log console để xem quá trình so sánh")
    print("   - Xem độ tương đồng (similarity) có đạt ngưỡng không")
    print("   - Kiểm tra face encodings có được tạo đúng không")

if __name__ == "__main__":
    print("🔧 CÔNG CỤ DEBUG NHẬN DIỆN KHUÔN MẶT")
    print("="*50)
    
    # Debug database
    debug_face_database()
    
    # Gợi ý cải thiện
    suggest_improvements()
    
    # Nếu có 2 ảnh để so sánh
    import sys
    if len(sys.argv) >= 3:
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        debug_face_comparison(image1, image2)
