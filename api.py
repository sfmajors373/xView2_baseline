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

app = FastAPI(title='Damage Assessment')

@app.post("/damage-assessment/", tags=["Damage Assessment"])
async def damage_assessment(png_pre: UploadFile = File(...), png_post: UploadFile = File(...), label_json: UploadFile = File(...)):
    os.mkdir("tmp_file_store")
    os.mkdir("tmp_file_store/output_polygons")
    # Later add the localization command also
    # python3 ./process_data_inference.py --input_img "$disaster_post_file" --label_path "$label_temp"/"${input_image%.*}".json --output_dir "$inference_base"/output_polygons --output_csv "$inference_base"/output.csv >> "$LOGFILE" 2>&1
    process_data_inference --input_img png_pre --label_path label_json --ouput_dir "tmp_file_store/output_polygons" --output_csv "tmp_file_store/output.csv"
    # python3 ./damage_inference.py --test_data "$inference_base"/output_polygons --test_csv "$inference_base"/output.csv --model_weights "$classification_weights" --output_json /tmp/inference/classification_inference.json >> "$LOGFILE" 2>&1
    damage_inference --test_data "tmp_file_store/output_polygons" --test_csv "tmp_file_store/output.csv" --model_weights "/model/modelweights/-saved-model-99-0.32.hdf5" --ouput_json "tmp_file_store/classification_inference.json"
    # python3 "$XBDIR"/utils/combine_jsons.py --polys "$label_temp"/"${input_image%.*}".json --classes /tmp/inference/classification_inference.json --output "$inference_base/inference.json" >> "$LOGFILE" 2>&1
    combine_jsons --polys label_json --classes "tmp_file_store/classification_inference.json" --output "tmp_file_store/inference.json"
    # python3 "$XBDIR"/utils/inference_image_output.py --input "$inference_base"/inference.json --output "$output_file"  >> "$LOGFILE" 2>&1
    inference_image_output --input "tmp_file_store/inference.json" --output "return.png"
    shutil.rmtree("tmp_file_store")
    img = Image.open("return.png", mode='r')
    img_byte_arr = io.BytesIO()
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/", tags=['Health Check'])
async def root():
    return {'message': 'OK'}
