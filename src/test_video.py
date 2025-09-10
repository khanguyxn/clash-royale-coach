import cv2
from matplotlib.pylab import svd
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import pytesseract
import numpy as np


load_dotenv()
path = "data/raw/test_video_1.MP4"
output_path = "data/labels/annotated_video.mp4"

def read_image(img):
     config = "--psm 13 -c tessedit_char_whitelist=0123456789"
     #img = Image.open(img)
     img = preprocess(img)
     text = pytesseract.image_to_string(img, config = config)
     cv2.imshow("video", img)
     print(text.strip())
     return text.strip()

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.resize(img, None, fx = 5, fy = 5, interpolation=cv2.INTER_CUBIC)
    return(img)


def frame_reader(path, output_path, model):
    #grab the video based on file path
    index = 0

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

    already_mapped = False

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
        #print(f'we have this many detections: {len(result)}')
        
        for detection in result:
            box = detection.boxes
            class_index = int(box.cls)
            class_name = result.names[class_index]

            if class_name == "elixir bar" and index % 5 ==0:
                 if already_mapped == False:   
                    x1, y1, x2, y2 = map(int, (box.xyxy[0].tolist()))
                    print("mapping again!")
                    already_mapped = True
                 img = result.orig_img
                 elixir_region = img[y1+4:y2-30, x1+65:x2-80]
                 print(f'Elixir count: {read_image(elixir_region)}')



        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        index += 1
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



#frame_reader(path, output_path, model)










        







