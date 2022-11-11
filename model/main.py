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

test_data = './tmp_file_store/output_polygons'
test_csv = './tmp_file_store/output.csv'
output_json_path = './tmp_file_store/classification_inference.json'

BATCH_SIZE = 64

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = 'destroyed'
damage_intensity_encoding[2] = 'major-damage'
damage_intensity_encoding[1] = 'minor-damage'
damage_intensity_encoding[0] = 'no-damage'


app = FastAPI(title="classification")

@app.get("/damage-classification/", tags=['Damage Classification'])
async def classification():
    print('*********************** MADE IT TO CLASSIFICATION *********************')

    # classify using nvidia_model (triton)
    df = pd.read_csv(test_csv)

    test_gen = create_generator(df, test_data, output_json_path)
    test_gen.reset()

    samples = df["uuid"].count()

    steps = np.ceil(samples/BATCH_SIZE)
    # steps_int = math.trunc(steps)

    predictions = []
    step = 0
    print('************************** BATCHING **********************')
    for x_batch in test_gen:
        print('!!!!!!!!! STEP COUNT: ', step)
        if step <= steps:   
            if x_batch.shape[0] < BATCH_SIZE:
                while x_batch.shape[0] < BATCH_SIZE:
                    x = np.zeros((1, 128, 128, 3), dtype=np.float32)
                    x_batch = np.concatenate((x_batch, x))
            preds = model.predict(x_batch)
            print('!!!!!!!!!!!!!! PREDS: ', preds.shape)
            for i in range(preds.shape[0]):
                predictions.append(preds[i])
            step += 1
        else:
            break

    # print('************************* PREDICTION SHAPE: ', predictions.shape)

    # for step in range(int(steps)):
    #     x_batch = test_gen.take(step)
    #     preds = model.predict(x_batch)
    #     predictions.append(predictions)

    #predictions = model.predict(test_gen)
    #print('####################### PREDICTIONS: ', len(predictions))

    print("***************************************** HAVE I FINISHED PREDICTING THINGS YET?????? ******************************")

    predicted_indices = np.argmax(predictions, axis=1)
    predictions_json = dict()

    for i in range(samples):
        filename_raw = test_gen.filenames[i]
        filename = filename_raw.split(".")[0]
        print('************************* PREDICTION: ', predicted_indices[i])
        predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]

    with open(output_json_path, 'w') as outfile:
        json.dump(predictions_json, outfile)

    return 1
