
import requests
import json
import os  # Importing the os module for handling file paths and directories
import sqlite3  # Importing the sqlite3 module to interact with the SQLite database
import uuid  # Importing the uuid module to generate unique identifiers
from fastapi import FastAPI, UploadFile, File, HTTPException, Form  # Importing FastAPI components
from fastapi.responses import FileResponse, JSONResponse  # Importing response types from FastAPI
import pandas as pd  # Importing pandas for data manipulation and saving to Excel
from text_extraction import pdf_to_images, rotate_and_save_image_if_needed, ocr_to_csv, process_dataframe  # Importing custom functions for text extraction
from summarize import pdf_to_corrected_images, extract_text_from_images, question_text, json_to_dataframe  # Importing custom functions for summarization

app = FastAPI()  # Initializing a FastAPI application
# Your API Key here (Consider using environment variables for security)
API_KEY = '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'

# SQLite database setup
DATABASE = 'pdf_processing.db'         # Path to the SQLite database file

def init_db():
    """Initialize the database by creating the necessary table if it doesn't exist."""
    with sqlite3.connect(DATABASE) as conn:  # Connect to the SQLite database
        cursor = conn.cursor()  # Create a cursor object to execute SQL commands
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                filename TEXT UNIQUE,  
                upload_path TEXT, 
                images_path TEXT, 
                text_path TEXT, 
                output_xlsx_path TEXT  
            )
        ''')  # SQL command to create the pdf_files table
        conn.commit()  # Commit the transaction to the database

init_db()  # Call the init_db function to ensure the database is set up

def save_pdf_to_db(filename, upload_path, images_path, text_path, output_xlsx_path):
    """Save the paths and filename related to a PDF file to the database."""
    with sqlite3.connect(DATABASE) as conn:  # Connect to the SQLite database
        cursor = conn.cursor()  # Create a cursor object to execute SQL commands
        cursor.execute('''
            INSERT OR IGNORE INTO pdf_files (filename, upload_path, images_path, text_path, output_xlsx_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, upload_path, images_path, text_path, output_xlsx_path))  # Insert the file paths into the database
        conn.commit()  # Commit the transaction to the database

def get_pdf_info(filename):
    """Retrieve information about a specific PDF file from the database."""
    with sqlite3.connect(DATABASE) as conn:  # Connect to the SQLite database
        cursor = conn.cursor()  # Create a cursor object to execute SQL commands
        cursor.execute('''
            SELECT * FROM pdf_files WHERE filename = ?
        ''', (filename,))  # Select the record for the given filename
        return cursor.fetchone()  # Return the record as a tuple
from summarize import pdf_to_corrected_images  # Importing the function from summarize.py

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
async def upload_pdf(file: UploadFile = File(...), action: str = Form(...)):
    """Endpoint to upload a PDF, process it, and store the results."""
    if not(file.filename.endswith(".pdf") or file.filename.endswith(".PDF")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")  # Validate the file type

    unique_id = str(uuid.uuid4().hex)  # Generate a unique ID for the upload
    base_dir = os.path.join("uploads", unique_id)  # Define the base directory for storing files related to this upload
    os.makedirs(base_dir, exist_ok=True)  # Create the base directory if it doesn't exist

    # Define paths for the uploaded file, images, text outputs, and final Excel file
    file_path = os.path.join(base_dir, file.filename)  # Path to save the uploaded PDF
    text_output_folder = os.path.join(base_dir, "texts")  # Directory to save the extracted text files
    output_xlsx_path = os.path.join(base_dir, "output.xlsx")  # Path to save the final Excel file

    # Save the uploaded PDF file to the defined file path
    with open(file_path, "wb") as f:  # Open the file in write-binary mode
        f.write(file.file.read())  # Write the uploaded content to the file

    try:
        if action == "text_extraction":
            # Convert PDF to corrected images using the pdf_to_corrected_images function
            image_paths = pdf_to_corrected_images(file_path)  # Convert the PDF to images and store them

            # Perform OCR on each image and save the results to CSV files
            csv_file_paths = []  # List to store paths to CSV files
            for image_path in image_paths:  # Iterate through each image path
                csv_file_path = image_path.replace(".jpg", ".csv")  # Define the CSV file path
                ocr_to_csv(image_path, csv_file_path)  # Perform OCR and save the result to a CSV file
                csv_file_paths.append(csv_file_path)  # Add the CSV file path to the list

            # Convert the CSV files to an Excel file with multiple sheets
            with pd.ExcelWriter(output_xlsx_path) as writer:  # Create an Excel writer object
                for csv_file_path in csv_file_paths:  # Iterate through each CSV file path
                    df = pd.read_csv(csv_file_path, header=None)  # Read the CSV file into a DataFrame
                    df = process_dataframe(df)  # Process the DataFrame to clean and structure the data
                    sheet_name = os.path.basename(csv_file_path).replace(".csv", "")  # Determine the sheet name from the CSV file name
                    df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)  # Write the DataFrame to the Excel sheet

            # Save the paths and filename to the database
            save_pdf_to_db(file.filename, file_path, os.path.dirname(image_paths[0]), text_output_folder, output_xlsx_path)  # Save file paths to the database

            return FileResponse(
                output_xlsx_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="output.xlsx"
            ) # Return a JSON response with a download link

        elif action == "summarized_extraction":
            # Convert PDF to corrected images using the pdf_to_corrected_images function
            image_paths = pdf_to_corrected_images(file_path)  # Convert the PDF to images and store them

            # Extract text from the images
            text_output_folder=extract_text_from_images(image_paths, output_folder=text_output_folder)  # Extract text from images and save it

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
            ])  # DataFrame to store all extracted data

            for filename in os.listdir(text_output_folder):  # Iterate through all text files in the output folder
                if filename.endswith(".txt"):  # Only process text files
                    file_path = os.path.join(text_output_folder, filename)  # Get the full path to the text file
                    with open(file_path, "r", encoding="utf-8") as f:  # Open the text file in read mode
                        text_content = f.read()  # Read the content of the text file
                    extraction_json = question_text(text_content, query_template, API_KEY)  # Extract data using the template
                    if extraction_json:  # If extraction was successful
                        df = json_to_dataframe(extraction_json)  # Convert the JSON extraction to a DataFrame
                        if df is not None:  # If the DataFrame is valid
                            all_data = pd.concat([all_data, df], ignore_index=True)  # Concatenate the DataFrame to all_data

            all_data.to_excel(output_xlsx_path, index=False)  # Save the extracted data to an Excel file

            # Save paths to the database
            save_pdf_to_db(file.filename, file_path, os.path.dirname(image_paths[0]), text_output_folder, output_xlsx_path)  # Save file paths to the database

            return FileResponse(
                output_xlsx_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="output.xlsx"
            ) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle any exceptions and return a 500 error
