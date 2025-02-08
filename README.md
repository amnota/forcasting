# Demand Forecasting API

นี่คือ Backend API สำหรับแอป Demand Forecasting ที่สร้างด้วย **FastAPI**  
API นี้รองรับการอัปโหลดไฟล์ CSV ที่มี historical sales data, ทำ data cleaning และใช้ ML model เพื่อพยากรณ์ยอดขาย

## คุณสมบัติ
- **CSV File Upload:** อัปโหลดไฟล์ CSV ที่มีข้อมูลยอดขาย
- **Data Cleaning:** ลบข้อมูลที่หายไป (null values)
- **Forecasting:** ใช้ ML model (จากไฟล์ `model.pkl`) เพื่อทำการพยากรณ์
- **Swagger UI:** เอกสาร API อัตโนมัติที่ `/docs`

## โครงสร้างโปรเจกต์
