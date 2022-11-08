from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
import math
import random
import json
from sys import exit
import cv2
import io
import os
import time
import datetime
import shutil
import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
import process_data_inference
import create_generator
from utils import combine_jsons
from utils import inference_image_output
from nvidia_model import DamageClassificationModel

triton_url = 'triton:8000'

model = DamageClassificationModel(triton_url)

test_data = '../tmp_file_store/output_polygons'
test_csv = '../tmp_file_store/output.csv'
output_json_path = '../tmp_file_store/classification_inference.json'


app = FastAPI(title="classification")

@app.post("/damage-classification", tags=['Damage Classification'])
async def classification():
    # process data for classification
    os.system('python3 ./model/process_data_inference.py --input_img "./tmp_file_store/input_files/png_post.png" --label_path "./tmp_file_store/localization.json" --output_dir "tmp_file_store/output_polygons" --output_csv "tmp_file_store/output.csv"')

    # classify
    # os.system('python3 ./model/damage_inference.py --test_data "tmp_file_store/output_polygons" --test_csv "tmp_file_store/output.csv" --model_weights "./model/model_weights/-saved-model-99-0.32.hdf5" --output_json "tmp_file_store/classification_inference.json"')

    # classify using nvidia_model (triton)
    df = pd.read_csv(test_csv)

    test_gen = create_generator(df, test_data, output_json_path)
    test_gen.reset()

    samples = df["uuid"].count()

    steps = np.ceil(samples/BATCH_SIZE)

    predictions = model.predict(test_gen, verbose=1)

    predicted_indices = np.argmax(predictions, axis=1)
    predictions_json = dict()

    for i in range(samples):
        filename_raw = test_gen.filenames[i]
        filename = filename_raw.split(".")[0]
        predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]

    with open(output_json_path, 'w') as outfile:
        json.dump(predictions_json, outfile)

    return 1

@app.get("/", tags=["Health Check", "Damage Classification"])
async def root():
    return {"message": "OK"}
