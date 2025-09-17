from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("ppmg-burgas").project("alpr-yolov8")
dataset = project.version(4).download("yolov8")
print("✅ تم تنزيل النموذج بنجاح!")