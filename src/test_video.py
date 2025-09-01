import cv2
from matplotlib.pylab import svd
from roboflow import Roboflow
from ultralytics import YOLO
import random
import supervision as sv
import os
import time as timer
from dotenv import load_dotenv
import pytesseract

def read_image(img):
     #img = cv2.imread(file_path)
     img = preprocess(img)
     text = pytesseract.image_to_string(img)
     return text

def preprocess(img):
    #img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)[1]
    #img = cv2.medianBlur(img, 5)
    return(img)




load_dotenv()
path = "data/raw/test_video_1.MP4"
output_path = "data/labels/annotated_video.mp4"


def frame_reader(path, output_path, model):
    #grab the video based on file path
   

    cap = cv2.VideoCapture(path)
    frames = []
    time = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width,height))
    frame_skipper = 15 #display frame once every 15 frames

    for result in model.predict(
        path, 
        conf = .6,
        imgsz = 640,
        half = False,
        vid_stride = 5,
        stream = True,
        show = True,
        save_frames = True,
        verbose = False,

    ): 
        print(f'we have this many detections: {len(result)}')
        for detection in result:
            box = detection.boxes
            class_index = int(box.cls)
            class_name = result.names[class_index]

            if class_name == "elixir bar":
                 x1, y1, x2, y2 = map(int, (box.xyxy[0].tolist()))
                 img = result.orig_img
                 elixir_region = result.orig_img[y1:y2, x1:x2]
                 print(read_image(elixir_region))


        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pass
    

'''
#Use if you haven't trained the model yet
api_key = os.getenv('API_KEY')
rf = Roboflow(api_key)
project = rf.workspace("khang-nguyen-evva6").project("elixir-timer-eh65e")
version = project.version(5)
dataset = version.download("yolov12")

model = YOLO("yolo12n.pt")
results = model.train(
    data = f"{dataset.location}/data.yaml",
    epochs = 250
)
'''

model_path = "/opt/homebrew/runs/detect/train3/weights/best.pt"
model = YOLO(model_path)

frame_reader(path, output_path, model)









        







