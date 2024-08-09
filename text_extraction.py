import os  # For file and directory operations
import cv2  # For image processing
import numpy as np  # For array operations
import tensorflow as tf  # For machine learning operations
import pytesseract  # For OCR
import imutils  # For image manipulation
from pdf2image import convert_from_path  # For converting PDF pages to images
from paddleocr import PaddleOCR  # For OCR using PaddleOCR
import pandas as pd  # For data manipulation
import csv  # For CSV file operations




# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

def pdf_to_images(pdf_path, output_dir, dpi=500):
    print(f"Converting PDF to images from: {pdf_path}")
    
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    image_paths = []
    image_counter = 1
    
    # Save each page as an image
    for page in pages:
        filename = os.path.join(output_dir, f"page_{image_counter}.jpg")
        print(f"Saving image to: {filename}")
        page.save(filename, "JPEG")
        
        # Verify if the file was saved successfully
        if os.path.exists(filename):
            print(f"Image saved successfully at {filename}")
            image_paths.append(filename)
        else:
            raise FileNotFoundError(f"Failed to save image at {filename}")
        
        image_counter += 1
    
    return image_paths



def rotate_and_save_image_if_needed(image_path, output_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)
    rotation_angle = int(results["rotate"])

    if rotation_angle != 0:
        rotated = imutils.rotate_bound(image, angle=rotation_angle)
        cv2.imwrite(output_path, rotated)
        print(f"Rotated image saved to: {output_path}")
    else:
        cv2.imwrite(output_path, image)
        print(f"Image was already in correct orientation, saved without rotation to: {output_path}")

def ocr_to_csv(image_path, csv_file_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    image_height = image_cv.shape[0]
    image_width = image_cv.shape[1]
    output = ocr.ocr(image_path)[0]
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]
    image_boxes = image_cv.copy()
    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), (int(box[2][1]))), (0, 0, 255), 1)
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)
    horiz_boxes = []
    vert_boxes = []
    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height
        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    horiz_lines = np.sort(np.array(horiz_out))
    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    vert_lines = np.sort(np.array(vert_out))
    out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]
    unordered_boxes = []
    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])
    ordered_boxes = np.argsort(unordered_boxes)

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])
        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1, 0)))
        if inter == 0:
            return 0
        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
        return inter / float(box_1_area + box_2_area - inter)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if iou(resultant, the_box) > 0.1:
                    out_array[i][j] = texts[b]
    out_array = np.array(out_array)
    # Post-process to align columns correctly and remove incomplete duplicates
    out_array = align_columns(out_array)
    out_array = remove_incomplete_duplicates(out_array)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(out_array)
    print(f"Data successfully saved to {csv_file_path}")

def align_columns(out_array):
    df = pd.DataFrame(out_array)
    for i in range(1, df.shape[1]):
        for j in range(df.shape[0]):
            if df.iloc[j, i] and not df.iloc[j, i - 1]:
                df.iloc[j, i - 1] = df.iloc[j, i]
                df.iloc[j, i] = ""
    return df.values

def remove_incomplete_duplicates(out_array):
    df = pd.DataFrame(out_array)
    df_cleaned = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_cleaned = df_cleaned.drop(columns=[col])
    return df_cleaned.values

def process_dataframe(df):
    def remove_duplicates_in_row(row):
        return row.drop_duplicates(keep='first')
    df_processed = df.apply(remove_duplicates_in_row, axis=1)
    return df_processed

def csv_to_xlsx(csv_file_path, xlsx_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    df = process_dataframe(df)
    df.to_excel(xlsx_file_path, index=False, header=False)
    print(f"CSV data successfully converted to XLSX format at {xlsx_file_path}")

def special_cleaning_task(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ["0", "1", "2", "3", "4", "5"]
    df = df.drop(df.index[15:19])
    df = df.drop(df.columns[[4, 5]], axis=1)
    cleaned_csv_path = csv_file_path.replace(".csv", "_cleaned.csv")
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Special cleaned CSV saved at {cleaned_csv_path}")
    return cleaned_csv_path

