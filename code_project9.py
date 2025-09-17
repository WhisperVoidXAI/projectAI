# -*- coding: utf-8 -*-

from ultralytics import YOLO
import pytesseract
import cv2
import pandas as pd
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
#هان غير لا
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model_path = r"D:\MyDesktop\project_zaid\best.pt"
image_path = r"D:\MyDesktop\project_zaid\image_cars\cropped\my-jordanian-cars-all-found-in-dubai-v0-wrw3szx83s4d1.webp"
csv_path = r"D:\MyDesktop\project_zaid\vi.csv"

# تحميل نموذج YOLO
model = YOLO(model_path)
results = model.predict(image_path, conf=0.25, save=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

# قراءة الصورة
img = cv2.imread(image_path)
height, width = img.shape[:2]

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    # تكبير المربع حول اللوحة (padding 10%)
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(width, x2 + pad_x)
    y2_pad = min(height, y2 + pad_y)

    plate_img = img[y1_pad:y2_pad, x1_pad:x2_pad]

    # تحسين الصورة للـ OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    # OCR للنص
    raw_text = pytesseract.image_to_string(gray, lang='eng', config='--psm 8').strip()

    # إذا لم يتم التعرف على أرقام، نجرب اللوحة مقلوبة أفقيًا
    if not re.search(r'\d', raw_text):
        flipped_img = cv2.flip(gray, 1)
        raw_text = pytesseract.image_to_string(flipped_img, lang='eng', config='--psm 8').strip()

    # تنظيف النص: الإبقاء على الأرقام والشرطة فقط
    plate_text = re.sub(r'[^0-9-]', '', raw_text)

    # تقسيم إلى جزئين إذا كانت هناك شرطة
    parts = plate_text.split('-')
    if len(parts) == 2:
        left_part, right_part = parts
        # تعديل الجهة اليمنى: أقصى 5 أرقام
        right_part = right_part[:5]
        # تعديل الجهة اليسرى: إذا زاد عن 2 أرقام، حذف آخر رقم فقط
        if len(left_part) > 2:
            left_part = left_part[1:]  # حذف آخر رقم فقط
        plate_text = f"{left_part}-{right_part}" if left_part else right_part
    else:
        # إذا لا توجد شرطة، نحتفظ فقط بأقصى 5 أرقام
        plate_text = parts[0][:5]

    print(f"📄 رقم اللوحة المكتشف بعد التنظيف: {plate_text}")

    # البحث في ملف CSV عن المخالفات
    violation_text = "No Violation"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # إزالة أي فراغات بين الأرقام
        df['plate_number_clean'] = df['plate_number'].str.replace(" ", "")
        matched = df[df['plate_number_clean'] == plate_text]

        if not matched.empty:
            violation_text = "Violation"
            print("⚠️ تم العثور على المخالفات التالية:")
            for index, row in matched.iterrows():
                print(
                    f"نوع المخالفة: {row['violation_type']}, "
                    f"المبلغ: {row['amount']}, الحالة: {row['status']}"
                )
        else:
            print("✅ لا توجد مخالفات لهذه اللوحة.")
    else:
        print(f"❌ ملف CSV غير موجود: {csv_path}")

    # رسم المربع والرقم على الصورة
    cv2.rectangle(img, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
    cv2.putText(
        img, f"{plate_text} - {violation_text}",
        (x1_pad, y1_pad - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2
    )

# عرض الصورة النهائية
cv2.imshow("Detected Plates", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
