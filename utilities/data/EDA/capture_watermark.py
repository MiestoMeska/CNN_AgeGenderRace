import os
from PIL import Image
import pytesseract
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\MiestoMeska\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def check_watermark(filename, path_data, min_confidence=50, min_text_length=5):
    img_path = os.path.join(path_data, filename)
    try:
        img = Image.open(img_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        text_blocks = []
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word:
                try:
                    conf = int(data['conf'][i])
                    if conf >= min_confidence:
                        text_blocks.append(word)
                except ValueError:
                    continue  # skip if confidence is not a number

        text = " ".join(text_blocks).strip()
        has_text = len(text) >= min_text_length

        return filename, has_text, text

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return filename, True, "ERROR"

def capture_watermark(df_labels, path_data, max_workers=8, min_confidence=85, min_text_length=3):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(check_watermark, fname, path_data, min_confidence, min_text_length)
            for fname in df_labels['filename']
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Scanning images for watermarks"):
            results.append(f.result())

    results_df = pd.DataFrame(results, columns=['filename', 'has_watermark', 'watermark_text'])
    df_merged = df_labels.merge(results_df, on='filename')
    return df_merged
