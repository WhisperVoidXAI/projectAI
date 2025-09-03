from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features,
labels, test_size=0.33, random_state=42)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, preds))
new_sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
              0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
              0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
              0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
prediction = gnb.predict([new_sample])
print("Prediction:", label_names[prediction[0]])
new_samples = [
    [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
     0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
     0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
     0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],  # العينة 1
    [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
     0.1885, 0.05766, 0.2416, 0.2079, 1.495, 8.41, 0.003569, 0.01308,
     0.01443, 0.007033, 0.02058, 0.003231, 17.33, 20.18, 112.0, 857.1,
     0.1405, 0.209, 0.2112, 0.09886, 0.1859, 0.06118]   # العينة 2
]
predictions = gnb.predict(new_samples)
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} prediction: {label_names[pred]}")