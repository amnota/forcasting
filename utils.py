import pandas as pd
import io
from io import BytesIO

async def read_file(file):
    """
    อ่านไฟล์ CSV หรือ Excel และแปลงเป็น DataFrame
    """
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(await file.read()))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(await file.read()))
        else:
            return None, "File type not supported"
        return df, None
    except Exception as e:
        return None, str(e)
