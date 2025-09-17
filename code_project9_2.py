import cv2
from ultralytics import YOLO
import pytesseract
import pandas as pd
import os
import re
import numpy as np
import time
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
model_path = r"D:\MyDesktop\project_zaid\best.pt"
csv_path = r"D:\MyDesktop\project_zaid\vi.csv"
camera_index = 0
conf_threshold = 0.25

pad_left, pad_right, pad_top, pad_bottom = 20, 20, 5, 5
base_save_dir = os.path.join(os.getcwd(), "captures")
plates_dir = os.path.join(base_save_dir, "plates")
frames_dir = os.path.join(base_save_dir, "frames")
os.makedirs(plates_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

last_saved_time = {}
save_cooldown_seconds = 3

# ---------- تحميل النموذج و CSV ----------
model = YOLO(model_path)
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    df_csv['plate_number_clean'] = df_csv['plate_number'].astype(str).str.replace(" ", "", regex=False)
else:
    df_csv = None
    print("⚠️ ملف CSV غير موجود، سيتم عرض الألواح فقط دون فحص المخالفات.")

# ---------- فتح الكاميرا ----------
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError(f"تعذر فتح الكاميرا بالمؤشر {camera_index}")

print("الكاميرا تعمل. اضغط 'q' للخروج.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("فشل في قراءة الإطار من الكاميرا.")
            break

        h_img, w_img = frame.shape[:2]
        display_frame = frame.copy()

        # ----- كشف اللوحات -----
        results = model(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) and results[0].boxes is not None else np.array([])
        confs = results[0].boxes.conf.cpu().numpy() if len(results) and results[0].boxes is not None else np.array([])

        for idx, box in enumerate(boxes):
            score = float(confs[idx]) if idx < len(confs) else 1.0
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1_exp = max(x1 - pad_left, 0)
            y1_exp = max(y1 - pad_top, 0)
            x2_exp = min(x2 + pad_right, w_img - 1)
            y2_exp = min(y2 + pad_bottom, h_img - 1)

            plate_img = frame[y1_exp:y2_exp, x1_exp:x2_exp]

            # ----- OCR -----
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789-'
            raw_text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng').strip()
            plate_text = re.sub(r'[^0-9-]', '', raw_text)
            if '-' not in plate_text and len(plate_text) >= 6:
                plate_text = plate_text[:2] + '-' + plate_text[2:]

            display_label = plate_text if plate_text else "Unknown"

            # ----- فحص CSV -----
            violation_status = "No Violation"
            if plate_text and df_csv is not None:
                plate_text_clean = plate_text.replace(" ", "")
                matched = df_csv[df_csv['plate_number_clean'] == plate_text_clean]
                if not matched.empty:
                    violation_status = "Violation"
                    print(f"⚠️ تم العثور على مخالفات للوحة {plate_text}:")
                    for index, row in matched.iterrows():
                        print(f"  - نوع المخالفة: {row['violation_type']}, المبلغ: {row['amount']}, الحالة: {row['status']}")
                else:
                    print(f"✅ لا توجد مخالفات للوحة {plate_text}.")

            # ----- رسم المستطيل والنص على display_frame -----
            cv2.rectangle(display_frame, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{display_label} | {violation_status}", 
                        (x1_exp, y2_exp + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2, cv2.LINE_AA)

            # ----- حفظ اللوحة مع المستطيل والنص -----
            if plate_text:
                now_ts = time.time()
                last_ts = last_saved_time.get(plate_text, 0)
                if now_ts - last_ts >= save_cooldown_seconds:
                    safe_name = re.sub(r'[^0-9\-]', '_', plate_text)
                    plate_fname = f"{safe_name}_{int(now_ts)}.png"
                    frame_fname = f"{safe_name}_frame_{int(now_ts)}.png"

                    # قص اللوحة من display_frame (مع المستطيل والنص)
                    plate_img_display = display_frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                    cv2.imwrite(os.path.join(plates_dir, plate_fname), plate_img_display)
                    # حفظ الإطار الكامل
                    cv2.imwrite(os.path.join(frames_dir, frame_fname), display_frame)
                    last_saved_time[plate_text] = now_ts

                    print(f"✅ حفظت لوحة: {plate_text} -> {plate_fname}")
                    print(f"✅ حفظت إطار الفيديو -> {frame_fname}")

        # عرض الفيديو
        cv2.imshow("Plate Detector (press q to quit)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("تم الإغلاق.")
