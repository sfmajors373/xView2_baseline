from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import imageio
import cv2
from PIL import Image
import io
import os
import shutil
import json
from model import process_data_inference
from model import damage_inference
from utils import combine_jsons
from utils import inference_image_output
from overlay_output_to_image import submission_to_overlay_polys


triton_url = 'triton:8000'

app = FastAPI(title='Damage Localization')

@app.post('/damage-localization/', tags=['Damage Localization', 'Stage 2'])
async def damage_localization():
    # localization
    os.system('python3 ./spacenet/src/models/inference.py --input "./tmp_file_store/input_files/png_pre.png" --weights "./model/model_weights/localization.h5" --mean "./weights/mean.npy" --output "tmp_file_store/localization.json"')
