import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pdf2image import convert_from_path
import cv2
import pytesseract
import imutils
from PIL import Image
import requests
import json
import pandas as pd

app = FastAPI()

# Initialize the OCR model
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
API_KEY = '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'

def pdf_to_corrected_images(pdf_path, dpi=500):
    pages = convert_from_path(pdf_path, dpi)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_images")
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    for idx, page in enumerate(pages, start=1):
        image_path = os.path.join(output_dir, f"page_{idx}.jpg")
        page.save(image_path, "JPEG")
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)
        rotation_angle = int(results["rotate"])
        if rotation_angle != 0:
            rotated_image = imutils.rotate_bound(image, angle=rotation_angle)
            cv2.imwrite(image_path, rotated_image)
        image_paths.append(image_path)
    return image_paths

def extract_text_from_images(image_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image_path in image_paths:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        text_output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    return output_folder

def question_text(text_content, query, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": f"{query}\n\n{text_content}"
            }
        ],
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def json_to_dataframe(json_str):
    try:
        start_index = json_str.find('{')
        end_index = json_str.rfind('}')
        json_str = json_str[start_index:end_index+1]
        data = json.loads(json_str)
        return pd.DataFrame(data, index=[0])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        image_paths = pdf_to_corrected_images(file_path)
        text_output_folder = extract_text_from_images(image_paths, output_folder="extracted_texts")
        query_template ="""
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
    "Flag": "LIBERIA"
    "Actual_Port_of_Loading": "VALENCIA",
    "Actual_Port_of_Discharge": "VALENCIA",
    "Description": "01 X 20ST STC  720 BAGS SOUCHETS BIOLOGIQUES TIGERNUTUS FROM ORGANIC MANAGEMENT ACCORDING TO ECC REG.834/2007 ",
    "Type_Container": "34360.000",
    "Nbre_colis": "720",
    "Volume_march":  "50.000" ,
    "Unite_volume": "MTQ",
    "Nom_du_Fichier":  "CMM1.pdf"    (Here in the value fill the name of the pdf)
}
"""
        all_data = pd.DataFrame(columns=[
            "BL", "Shipper", "Consignee", "Notifie", "Vessel", "Voyage", "Flag",
            "Actual_Port_of_Loading", "Actual_Port_of_Discharge", "Description", "Type_Container", "Nbre_colis","Volume_march","Unite_volume","Nom_du_Fichier"
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