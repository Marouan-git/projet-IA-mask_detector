from fastapi import Body, FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
from joblib import load
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import sqlite3
import datetime
from sklearn.decomposition import PCA


import mediapipe as mp


def detect_faces(file):
    # load face detection model
    mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.3 # confidence threshold
    )
    
    
    
    new_file = np.frombuffer(file, dtype=np.uint8)
    
    image = cv2.imdecode(new_file, cv2.IMREAD_COLOR)
    
    image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.open(file)
    # image_input = np.array(image)
    width = image.shape[1]
    height = image.shape[0]
    results = mp_face.process(image_input)
    faces = []
    
    if results.detections:
        for face in results.detections:
            bbox = face.location_data.relative_bounding_box
            
            bbox_points = {
            "xmin" : int(bbox.xmin * width),
            "ymin" : int(bbox.ymin * height),
            "xmax" : int(bbox.width * width + bbox.xmin * width),
            "ymax" : int(bbox.height * height + bbox.ymin * height)
            }
            
            pil_img = Image.fromarray(image)
            crop = pil_img.crop((bbox_points["xmin"], bbox_points["ymin"], bbox_points["xmax"], bbox_points["ymax"]))
            crop_resized = crop.resize((80,80))
            crop_gray = crop_resized.convert('L')
            crop_np = np.reshape(np.array(crop_gray, dtype='uint16'), newshape=(80*80))
            faces.append(crop_np)

    return faces


model = load("rf_complet.joblib")


    
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("file.html", {"request": request})

@app.post("/")
async def create_upload_file(request: Request, file: UploadFile):
    rfile = file.file.read()
    faces = detect_faces(rfile)
    print(len(faces))
    i = 1
    faces_files = []
    pred = []
    for face in faces:
         
         pred.append(model.predict(face.reshape(1,-1))[0])
         img = Image.fromarray(np.reshape(np.uint8(face), (80,80)))
         img.save(f"static/face{i}.png")
         faces_files.append(f"face{i}.png")
         i += 1
         
    print(pred)
    print(faces_files)
    
    con = sqlite3.connect('mask_monitoring.db') 

    cur = con.cursor()

    # Create table
    cur.execute('''CREATE TABLE IF NOT EXISTS monitoring
                (date text, prediction text)''')
    # Insert a row of data
    for label in pred:
        date = datetime.datetime.today()
        print(date)
        cur.execute(f"INSERT INTO monitoring (date, prediction) VALUES ('{date}','{label}')")
        # Save (commit) the changes
        con.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    con.close()
    
    return templates.TemplateResponse("result.html", {"request": request, "pred": list(zip(pred, faces_files))})     

@app.get("/webcam/", response_class=HTMLResponse) 
async def webcam(request: Request)   :
       return templates.TemplateResponse("webcam.html", {"request": request}) 
   
@app.get("/video", response_class=HTMLResponse)
async def video(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})
   
# @app.post("/webcam/")
# async def predict_webcam(request: Request):
#     faces = detect_faces(file)
#     print(len(faces))
#     i = 1
#     faces_files = []
#     pred = []
#     for face in faces:
#          pred.append(model.predict(face.reshape(1,-1))[0])
#          img = Image.fromarray(np.reshape(np.uint8(face), (80,80)))
#          img.save(f"static/face{i}.png")
#          faces_files.append(f"face{i}.png")
#          i += 1
         
#     print(pred)
#     print(faces_files)
    
#     photo = request.form.__get__("photo")
#     print(photo)
    
#     return templates.TemplateResponse("result.html", {"request": request, "pred": list(zip([], []))})   

  

# @app.post("/api")
# async def process_image(request: Request):
#     img = request.form()
#     print(img)                  
    
#     return {"img": img}
