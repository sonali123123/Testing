# PDF Processing and Data Extraction API

This project provides a FastAPI-based web service for processing PDF files, extracting text, and summarizing the content.

## Features

- PDF upload and processing
- Text extraction from PDF pages
- Summarized extraction of key information
- Database storage of processed file information
- Excel output generation

## Installation

1. Clone the repository
2. Install the required dependencies:

pip install -r requirements.txt

### Additional Tools

Tesseract-OCR: Download and add the executable file path to the environment variable.

Poppler: Download from [Poppler](https://poppler.freedesktop.org/releases.html), and add the lib and bin paths to the environment variables.

cuDNN: Download from [NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive), and add the bin and lib paths to the environment variables.

## Usage

1. Start the server: uvicorn app:app --reload
   
2. Use the `/upload_pdf/` endpoint to upload a PDF file and specify the action ("text_extraction" or "summarized_extraction")

3. The API will process the PDF and return either a JSON response with a download link or a direct file response with the extracted data in Excel format
   

## API Endpoints

- POST `/upload_pdf/`: Upload and process a PDF file
- GET `/download/{unique_id}`: Download the processed Excel file

## Configuration

- Set your OpenAI API key in the `API_KEY` variable in `app.py`
- Adjust the `DATABASE` path in `app.py` if needed

## File Structure

- `app.py`: Main FastAPI application
- `text_extraction.py`: Functions for PDF to image conversion and OCR
- `summarize.py`: Functions for text summarization and data extraction


## Database

The project uses SQLite to store information about processed PDF files. The database schema includes:

- `id`: Unique identifier
- `filename`: Name of the processed PDF file
- `upload_path`: Path where the uploaded PDF is stored
- `images_path`: Path where extracted images are stored
- `text_path`: Path where extracted text is stored
- `output_xlsx_path`: Path of the generated Excel file
   
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
