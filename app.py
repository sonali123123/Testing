import os
import sqlite3
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import pandas as pd
from text_extraction import pdf_to_images, rotate_and_save_image_if_needed, ocr_to_csv, process_dataframe
from summarize import pdf_to_corrected_images, extract_text_from_images, question_text, json_to_dataframe

app = FastAPI()

# Your API Key here (Consider using environment variables for security)
API_KEY = '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'

# SQLite database setup
DATABASE = 'pdf_processing.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                upload_path TEXT,
                images_path TEXT,
                text_path TEXT
            )
        ''')
        conn.commit()

init_db()

def save_pdf_to_db(filename, upload_path, images_path, text_path):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO pdf_files (filename, upload_path, images_path, text_path)
            VALUES (?, ?, ?, ?)
        ''', (filename, upload_path, images_path, text_path))
        conn.commit()

def get_pdf_info(filename):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM pdf_files WHERE filename = ?
        ''', (filename,))
        return cursor.fetchone()

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), action: str = Form(...)):
    if not(file.filename.endswith(".pdf") or file.filename.endswith(".PDF")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Use a different directory for images
    images_dir = os.path.join(upload_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Use a different directory for text outputs
    text_output_folder = os.path.join(upload_dir, "texts")
    os.makedirs(text_output_folder, exist_ok=True)

    if action == "text_extraction":
        try:
            # Convert PDF to images
            image_paths = pdf_to_images(file_path, images_dir)

            # Perform OCR and save to CSV
            csv_file_paths = []
            for image_path in image_paths:
                rotated_image_path = image_path.replace(".jpg", "_rotated.jpg")
                rotate_and_save_image_if_needed(image_path, rotated_image_path)
                csv_file_path = rotated_image_path.replace(".jpg", ".csv")
                ocr_to_csv(rotated_image_path, csv_file_path)
                csv_file_paths.append(csv_file_path)

            # Convert CSVs to XLSX
            xlsx_file_path = os.path.join(upload_dir, "output.xlsx")
            with pd.ExcelWriter(xlsx_file_path) as writer:
                for csv_file_path in csv_file_paths:
                    df = pd.read_csv(csv_file_path, header=None)
                    df = process_dataframe(df)
                    sheet_name = os.path.basename(csv_file_path).replace(".csv", "")
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)

            return FileResponse(xlsx_file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="output.xlsx")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif action == "summarized_extraction":
        try:
            image_paths = pdf_to_images(file_path, images_dir)
            extract_text_from_images(image_paths, output_folder=text_output_folder)

            query_template = """
I am providing you an excel file, multiple sheets are there. Read those sheets and
extract the following information:

BL
Shipper
Consignee
Notifie
Vessel
Voyage
Flag
Actual_Port_of_Loading
Actual_Port_of_Discharge
Description
Type_Container
Nbre_colis
Poids_march
Nbre_contener
Volume_march
Unite_volume
Nom_du_Fichier

The data should be formatted in JSON as follows:
{
    "BL": "<value>",
    "Shipper": "<value>",
    "Consignee": "<value>",
    "Notifie": "<value>",
    "Vessel": "<value>",
    "Voyage": "<value>",
    "Flag": "<value>",
    "Actual_Port_of_Loading": "<value>",
    "Actual_Port_of_Discharge": "<value>",
    "Type_Container": "<value>",
    "Volume_march": "<value>",
    "Unite_volume": "<value>",
    "Nom_du_Fichier": "<value>"
}

Below is one of the example that I am providing you. Try to extract information from all images in this way:
For Example:
{
    "BL": "BBD0102698",
    "Shipper": "FASO AGRI NATURE BONHEUR-VILLE, SRCTEUR 31 01 BP 6629 OUAGADOUGOU01 BURKINA FASO",
    "Consignee": "TIGERNUTS TRADERS, SL-CV1818L AV.POBLA DE VALBNA, 39 VALENCIA SPAIN",
    "Notifie": "TIGERNUTS TRADERS, SL-CV1818L AV.POBLA DE VALBNA, 39 VALENCIA SPAIN ",
    "Vessel": "CALYPSO",
    "Voyage": "OMY34N1MA",
    "Flag": "LIBERIA",
    "Actual_Port_of_Loading": "VALENCIA",
    "Actual_Port_of_Discharge": "VALENCIA",
    "Description": "01 X 20ST STC  720 BAGS SOUCHETS BIOLOGIQUES TIGERNUTUS FROM ORGANIC MANAGEMENT ACCORDING TO ECC REG.834/2007 ",
    "Type_Container": "34360.000",
    "Nbre_colis": "720",
    "Volume_march": "50.000",
    "Unite_volume": "MTQ",
    "Nom_du_Fichier": Use the name of the pdf here. For example if the name of pdf is CMM1 then use this.
}
"""
            all_data = pd.DataFrame(columns=[
                "BL", "Shipper", "Consignee", "Notifie", "Vessel", "Voyage", "Flag", 
                "Actual_Port_of_Loading", "Actual_Port_of_Discharge", "Description", 
                "Type_Container", "Nbre_colis", "Volume_march", "Unite_volume", "Nom_du_Fichier"
            ])

            for filename in os.listdir(text_output_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(text_output_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    extraction_json = question_text(text_content, query_template, API_KEY)
                    if extraction_json:
                        df = json_to_dataframe(extraction_json)
                        if df is not None:
                            all_data = pd.concat([all_data, df], ignore_index=True)

            output_xlsx_path = os.path.join(upload_dir, "output.xlsx")
            all_data.to_excel(output_xlsx_path, index=False)
            return FileResponse(output_xlsx_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="output.xlsx")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
