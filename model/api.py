from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import io
import os
import shutil
import model/process_data_inference
import model/damage_inference
import utils/combine_jsons
import utils/inference_image_output

triton_url = 'triton:8000'

model_path = ''
model = 

app = FastAPI(title="classification")

@app.post("/damage-classification", tags=['Damage Classification'])
async def classification():
    # process data for classification
    os.system('python3 ./model/process_data_inference.py --input_img "./tmp_file_store/input_files/png_post.png" --label_path "./tmp_file_store/localization.json" --output_dir "tmp_file_store/output_polygons" --output_csv "tmp_file_store/output.csv"')

    # classify
    os.system('python3 ./model/damage_inference.py --test_data "tmp_file_store/output_polygons" --test_csv "tmp_file_store/output.csv" --model_weights "./model/model_weights/-saved-model-99-0.32.hdf5" --output_json "tmp_file_store/classification_inference.json"')

    return
