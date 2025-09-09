# -----------------------------------------------
# 1️⃣ استيراد المكتبات الأساسية
# -----------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree

import lime
import lime.lime_tabular

plt.style.use('fivethirtyeight')

# -----------------------------------------------
# 2️⃣ تحميل البيانات ومعالجتها
# -----------------------------------------------
data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')

# تحويل "Not Available" إلى NaN
data.replace({'Not Available': np.nan}, inplace=True)

# الأعمدة الرقمية المهمة
numeric_keywords = ['ft²', 'kWh', 'kBtu', 'therms', 'Metric Tons CO2e', 'gal', 'Score']
numeric_columns = [col for col in data.columns if any(k in col for k in numeric_keywords)]

# تحويل القيم الرقمية
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
    data[col].fillna(data[col].mean(), inplace=True)

# التعامل مع القيم المتطرفة
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.clip(data[col], lower, upper)

# الأعمدة الزمنية
if 'Release Date' in data.columns:
    data['Release Date'] = pd.to_datetime(data['Release Date'], errors='coerce')
    median_date = data['Release Date'].median()
    data['Release Date'].fillna(median_date, inplace=True)

# الأعمدة الجغرافية
geo_columns = ['Latitude', 'Longitude']
for col in geo_columns:
    if col in data.columns:
        data.dropna(subset=[col], inplace=True)

# الأعمدة النصية
text_columns = ['Property Name', 'Primary Property Type - Self Selected']
for col in text_columns:
    if col in data.columns:
        data[col].fillna('Unknown', inplace=True)

print("معالجة البيانات مكتملة.")
print("عدد القيم المفقودة لكل عمود:")
print(data.isna().sum())

# -----------------------------------------------
# 3️⃣ استكشاف البيانات ورسم توزيعاتها
# -----------------------------------------------
plt.hist(data['ENERGY STAR Score'].dropna(), bins=100, edgecolor='k')
plt.xlabel('ENERGY STAR Score')
plt.ylabel('Number of Buildings')
plt.title('Energy Star Score Distribution')
plt.show()

# -----------------------------------------------
# 4️⃣ تجهيز الميزات والبيانات التدريبية
# -----------------------------------------------
features = data.copy()

# الأعمدة الرقمية
numeric_subset = data.select_dtypes('number').copy()
epsilon = 1e-6
for col in numeric_subset.columns:
    if col != 'ENERGY STAR Score':
        numeric_subset['log_' + col] = np.log1p(numeric_subset[col].where(numeric_subset[col] > 0, epsilon))

# الأعمدة الفئوية
categorical_subset = pd.get_dummies(data[['Borough', 'Largest Property Use Type']], drop_first=True)

# دمج البيانات الرقمية والفئوية
features = pd.concat([numeric_subset, categorical_subset], axis=1)

# العمود الهدف
target_column = 'ENERGY STAR Score'
targets = data[target_column]

# إزالة العمود الهدف من الميزات
X = features.drop(columns=[target_column], errors='ignore')

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3, random_state=42)

# -----------------------------------------------
# 5️⃣ ملء القيم المفقودة وتطبيع البيانات
# -----------------------------------------------
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(X_train_imputed)
X_train_scaled = scaler.transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# -----------------------------------------------
# 6️⃣ تدريب نموذج Gradient Boosting
# -----------------------------------------------
final_model = GradientBoostingRegressor(
    loss='absolute_error',
    max_depth=5,
    max_features=None,
    min_samples_leaf=8,
    min_samples_split=6,
    n_estimators=500,
    random_state=42
)
final_model.fit(X_train_scaled, y_train)

# -----------------------------------------------
# 7️⃣ استخراج أهمية الميزات ورسمها
# -----------------------------------------------
importances = final_model.feature_importances_
feature_list = list(X_train.columns)

feature_results = pd.DataFrame({
    'feature': feature_list,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("أهم 10 ميزات:")
print(feature_results.head(10))

plt.figure(figsize=(10,6))
plt.barh(feature_results.head(10)['feature'][::-1], feature_results.head(10)['importance'][::-1], color='skyblue')
plt.xlabel("أهمية الميزات")
plt.ylabel("الميزات")
plt.title("أهم 10 ميزات لنموذج GradientBoostingRegressor")
plt.show()

# -----------------------------------------------
# 8️⃣ تصدير شجرة فردية كصورة
# -----------------------------------------------
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

single_tree = final_model.estimators_[105][0]
dot_file_path = os.path.join(output_dir, "tree.dot")
tree.export_graphviz(
    single_tree,
    out_file=dot_file_path,
    feature_names=feature_list,
    filled=True,
    rounded=True,
    special_characters=True
)

# تحويل .dot إلى PNG باستخدام الأمر dot
os.system(f"dot -Tpng {dot_file_path} -o {os.path.join(output_dir,'tree.png')}")

# عرض الصورة
import matplotlib.image as mpimg
img = mpimg.imread(os.path.join(output_dir,'tree.png'))
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis('off')
plt.show()

# -----------------------------------------------
# 9️⃣ العثور على العينة ذات أكبر خطأ
# -----------------------------------------------
model_pred = final_model.predict(X_test_scaled)
residuals = np.abs(model_pred - y_test)
idx_max = np.argmax(residuals)
wrong = X_test_scaled[idx_max, :]

print('Prediction: %0.4f' % model_pred[idx_max])
print('Actual Value: %0.4f' % y_test.iloc[idx_max])
print('Index of max residual:', idx_max)

# -----------------------------------------------
# 🔟 تفسير العينة ذات أكبر خطأ باستخدام LIME
# -----------------------------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    training_labels=y_train,
    mode='regression',
    feature_names=feature_list
)

exp = explainer.explain_instance(
    data_row=wrong,
    predict_fn=final_model.predict
)

# عرض الرسم
fig = exp.as_pyplot_figure()
plt.show()
exp.show_in_notebook()
