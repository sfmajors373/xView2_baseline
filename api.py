from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import imageio
import time
import cv2
from PIL import Image
import io
import os
import shutil
import json
import requests
from model import process_data_inference
from model import damage_inference
from utils import combine_jsons
from utils import inference_image_output
from overlay_output_to_image import submission_to_overlay_polys

app = FastAPI(title='Damage Assessment')

@app.post("/damage-assessment", tags=["Damage Assessment"])
async def damage_assessment(png_pre: UploadFile = File(...), png_post: UploadFile = File(...), label_json: UploadFile = File(...)):
    # Set up tmp file tree to store temporary images
    if "tmp_file_store" in os.listdir('.'):
        shutil.rmtree("tmp_file_store")
    os.mkdir("tmp_file_store")
    os.mkdir("tmp_file_store/output_polygons")
    os.mkdir("tmp_file_store/input_files")
    os.mkdir("tmp_file_store/tmp_imgs")

    # Save inputs
    png_pre_contents = await png_pre.read()
    png_pre_nparr = np.fromstring(png_pre_contents, np.uint8)
    png_pre_img = cv2.imdecode(png_pre_nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("./tmp_file_store/input_files/png_pre.png", png_pre_img)

    png_post_contents = await png_post.read()
    png_post_nparr = np.fromstring(png_post_contents, np.uint8)
    png_post_img = cv2.imdecode(png_post_nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("./tmp_file_store/input_files/png_post.png", png_post_img)

    label_json_contents = await label_json.read()
    label_json_string = label_json_contents.decode("utf-8")
    with open("./tmp_file_store/input_files/label_json.json", "w") as outfile:
        outfile.write(label_json_string)


    # run the scripts

    # localization
    os.system('python3 ./spacenet/src/models/inference.py --input "./tmp_file_store/input_files/png_pre.png" --weights "./model/model_weights/localization.h5" --mean "./weights/mean.npy" --output "tmp_file_store/localization.json"')

    # process data for classification
    os.system('python3 ./model/process_data_inference.py --input_img "./tmp_file_store/input_files/png_post.png" --label_path "./tmp_file_store/localization.json" --output_dir "tmp_file_store/output_polygons" --output_csv "tmp_file_store/output.csv"')

    # classify
    #os.system('python3 ./model/damage_inference.py --test_data "tmp_file_store/output_polygons" --test_csv "tmp_file_store/output.csv" --model_weights "./model/model_weights/-saved-model-99-0.32.hdf5" --output_json "tmp_file_store/classification_inference.json"')

    # classify with the other api
    print('*************** REQUESTING ***************')
    predicts = requests.get('http://damage-classification:8004/damage-classification/')
    print('*************** DONE REQUESTING ***************')

    # while predicts != 1:
    #     time.sleep(2)
    # print('PREDICTS')
    # print(predicts)

    # Combining the predicted polygons with the predicted labels, based off a UUID generated during the localization inference stage 
    os.system('python3 ./utils/combine_jsons.py --polys "./tmp_file_store/localization.json" --classes "tmp_file_store/classification_inference.json" --output "tmp_file_store/inference.json"')

    # Make image for classification
    os.system('python3 ./utils/inference_image_output.py --input "tmp_file_store/inference.json" --output "tmp_file_store/tmp_imgs/classification.png"')
    
    # Combined image
    os.system('python3 ./overlay_output_to_image/submission_to_overlay_polys.py --image "tmp_file_store/input_files/png_post.png" --localization "tmp_file_store/tmp_imgs/classification.png" --damage "tmp_file_store/tmp_imgs/classification.png" --output "return.png"')

    # Convert image to bytes
    img = imageio.imread("return.png")
    _, png_img = cv2.imencode('.png', img)


    # Clean up the tmp file tree
    #shutil.rmtree("tmp_file_store")

    return StreamingResponse(io.BytesIO(png_img.tobytes()), media_type="image/png")

# @app.get("/", tags=['Health Check'])
# async def root():
#     return {'message': 'OK'}
