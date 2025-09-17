import pandas as pd
import random
n = 2000
violations = {
    "سرعة زائدة": 30,
    "اصطفاف مزدوج": 15,
    "استخدام الهاتف": 15,
    "تجاوز إشارة": 40,
    "عدم ارتداء حزام الأمان": 20
}
statuses = ["Paid", "Unpaid"]
def generate_plate():
    left = random.randint(10, 99)
    right = random.randint(1000, 9999)
    return f"{left}-{right}"
data = []
for _ in range(n):
    violation = random.choice(list(violations.keys()))
    amount = f"{violations[violation]} JD"
    row = {
        "plate_number": generate_plate(),
        "violation_type": violation,
        "amount": amount,
        "status": random.choice(statuses)
    }
    data.append(row)
df = pd.DataFrame(data)
output_file = "vi.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"تم إنشاء الملف: {output_file}")
