from utils import read_file 


from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

# พยายามโหลดโมเดลที่บันทึกไว้ในไฟล์ model.pkl
try:
    model = joblib.load("model.pkl")
    print("Loaded ML model from model.pkl")
except Exception as e:
    model = None
    print("Warning: ไม่พบ model.pkl หรือโหลดไม่ได้, จะใช้ dummy predictions แทน")

@app.get("/")
def index():
    return {"message": "Demand Forecasting API is running!"}

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    """
    รับไฟล์ CSV, ทำความสะอาดข้อมูล (ลบค่า null),
    ตรวจสอบว่ามีคอลัมน์ 'feature' อยู่ในข้อมูลหรือไม่,
    จากนั้นใช้ ML model เพื่อพยากรณ์ (หรือให้ dummy predictions หากไม่มี model)
    """
    try:
        # อ่านข้อมูลจากไฟล์ CSV ที่อัปโหลดเข้ามา
        data = await file.read()
        df = pd.read_csv(BytesIO(data))
        
        # Data cleaning: ลบข้อมูลที่เป็น NaN
        df.dropna(inplace=True)
        
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นอยู่หรือไม่
        required_column = "feature"  # เปลี่ยนชื่อตามข้อมูลจริงของคุณ
        if required_column not in df.columns:
            return {"error": f"CSV file ต้องมีคอลัมน์ '{required_column}'"}
        
        # หากมีโมเดลที่โหลดมาได้, ใช้โมเดลนั้นในการพยากรณ์
        if model is not None:
            predictions = model.predict(df[[required_column]])
            predictions_list = predictions.tolist()
        else:
            # ถ้าไม่มีโมเดล, คืนค่า dummy predictions
            predictions_list = [42] * len(df)
        
        return {"predictions": predictions_list}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
