import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# สร้างข้อมูลตัวอย่าง
data = {
    'feature': [1, 2, 3, 4, 5],
    'sales': [15, 30, 45, 60, 75]  # ข้อมูลยอดขายตัวอย่าง (คุณสามารถปรับเปลี่ยนได้)
}

df = pd.DataFrame(data)

# กำหนด Feature และ Target
X = df[['feature']]
y = df['sales']

# เทรนโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# บันทึกโมเดลเป็นไฟล์ model.pkl
joblib.dump(model, 'model.pkl')

print("Model saved as model.pkl")
