print("🔥 Loaded latest version of main.py!")

from utils import read_file
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# เพิ่ม CORS Middleware (รองรับ Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ควรเปลี่ยนเป็น ["https://your-frontend.com"] ใน production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# พยายามโหลดโมเดลที่บันทึกไว้ในไฟล์ model.pkl
try:
    model = joblib.load("model.pkl")
    logger.info("Loaded ML model from model.pkl")
except Exception as e:
    model = None
    logger.warning(f"Warning: ไม่พบ model.pkl หรือโหลดไม่ได้, จะใช้ dummy predictions แทน - {e}")

@app.get("/")
def index():
    return {"message": "Demand Forecasting API is running!"}

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    df, error = await read_file(file)
    if error:
        return {"error": error}

    df.dropna(inplace=True)

    # Features ที่ต้องใช้
    required_columns = ['past_sales', 'day_of_week', 'month', 'promotions', 'holidays', 'stock_level', 'customer_traffic']

    # 🔍 Debug: เช็คว่าคอลัมน์ที่อ่านได้ตรงกับที่ต้องการหรือไม่
    print("🔍 DEBUG: คอลัมน์ที่อ่านจาก CSV:", df.columns.tolist())
    # ✅ แก้ไข Indentation Error
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {"error": f"CSV file ต้องมีคอลัมน์ {missing_columns}"}

    try:
        if model is not None:
            predictions = model.predict(df[required_columns])
            df['forecast_sales'] = predictions

            # ✅ ตรวจสอบว่ามี actual_sales หรือไม่
            actual_sales = df.get('actual_sales')
            if actual_sales is not None and not actual_sales.isnull().all():
                df['error'] = abs(df['forecast_sales'] - actual_sales)
                forecast_accuracy = 100 - (df['error'].mean() / actual_sales.mean() * 100)
                overstock_risk = (df[df['forecast_sales'] > actual_sales].shape[0] / len(df)) * 100
                understock_risk = (df[df['forecast_sales'] < actual_sales].shape[0] / len(df)) * 100
            else:
                forecast_accuracy = 0  # เปลี่ยนจาก None เป็น 0
                overstock_risk = 0
                understock_risk = 0

            logger.info(f"Forecasting completed: Accuracy={forecast_accuracy:.2f}%, Overstock Risk={overstock_risk:.2f}%, Understock Risk={understock_risk:.2f}%")

            return {
                "predictions": df['forecast_sales'].tolist(),
                "forecast_accuracy": forecast_accuracy,
                "overstock_risk": overstock_risk,
                "understock_risk": understock_risk
            }
        else:
            logger.error("Model not loaded")
            return {"error": "Model not loaded"}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": f"Prediction failed: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
