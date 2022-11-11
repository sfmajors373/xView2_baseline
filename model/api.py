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
from nvidia_inference import create_generator
from nvidia_model import DamageClassificationModel
from utils import combine_jsons
from utils import inference_image_output

triton_url = 'triton:8000'

model = DamageClassificationModel(triton_url)

test_data = '../tmp_file_store/output_polygons'
test_csv = '../tmp_file_store/output.csv'
output_json_path = '../tmp_file_store/classification_inference.json'


#app = FastAPI(title="classification")

#@app.get("/damage-classification/", tags=['Damage Classification'])
async def classification():

    print('############################## Got to here! ###################################')

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
