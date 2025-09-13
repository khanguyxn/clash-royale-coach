import cv2
from matplotlib.pylab import svd
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import pytesseract
import ai_tester
from tkinter import *
import pandas as pd
import os



load_dotenv()
video_path = "data/raw/test_video_1.MP4"

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

#to show overall detections
def frame_reader(path, output_path, model): 
    #grab the video based on file path
    index = 0

   

    already_mapped = False

    for result in model.predict(
        path, 
        conf = .1,
        imgsz = (864, 480),
        iou = .999,
        half = False,
        vid_stride = 5,
        stream = True,
        show = True,
        verbose = False,
        augment = False

    ): 
        #print(f'we have this many detections: {len(result)}')
        
        for detection in result:
            box = detection.boxes
            class_index = int(box.cls)
            class_name = result.names[class_index]
            '''
            if class_name == "elixir bar" and index % 5 ==0:
                 
                 if already_mapped == False:   
                    x1, y1, x2, y2 = map(int, (box.xyxy[0].tolist()))
                    print("mapping again!")
                    already_mapped = True
                 img = result.orig_img
                 elixir_region = img[y1+4:y2-30, x1+65:x2-80]
                 print(f'Elixir count: {read_image(elixir_region)}')'''


        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        index += 1
        pass


#to be run when button is clicked 
iteration = 1
def scanFrame(img, hand_model, elixir_timer_model):
     enemy_troops = {
            "name" : [],
            "position" : [],
     }  #"name", "position"
     player_troops = {
          "name" : [],
          "position" : [],
     }#"name", "position"
     player_hand = {
          "curr_hand_name" : [],
     } #curr hand name
     #need time, elixir, try for opponent elixir...?

     #for the cards in hand: 
     for result in hand_model.predict(
          img,
          iou = .5,
          conf = .2,
          imgsz = (864, 480),
          show = True,
          verbose = False,
          save = True,
          project = "data/annotated/",
     ):
        for box in result.boxes:
            class_index = int(box.cls)
            class_name = result.names[class_index]
            print(f"class name is: {class_name}")
            player_hand["curr_hand_name"].append(class_name)

     ai_tester.runModel(str(player_hand))



def runVideo(video_path, hand_model, elixir_timer_model):
    cap = cv2.VideoCapture(video_path)
    while True:
         ret, frame = cap.read()
        
         if not ret:
              break
         cv2.imshow("Video", frame)

         key = cv2.waitKey(1) & 0xFF 
         if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break       
         elif key == ord('h'):
              scanFrame(frame, hand_model, elixir_timer_model)





'''
#Use if you haven't trained the model yet
#https://app.roboflow.com/khang-nguyen-evva6/cr_hand_cards-fojbu/models/cr_hand_cards-fojbu-instant-2
api_key = os.getenv('API_KEY')
rf = Roboflow(api_key)
project = rf.workspace("khang-nguyen-evva6").project("cr_hand_cards-fojbu")
version = project.version(1)
dataset = version.download("yolov12")

model = YOLO("yolo12n.pt")
results = model.train(
    data = f"{dataset.location}/data.yaml",
    epochs = 50,
)
'''


#ai_tester.runModel()


def main():
     hand_model_path = "/opt/homebrew/runs/detect/train8/weights/best.pt"
     hand_model = YOLO(hand_model_path)

     elixir_timer_path = "/opt/homebrew/runs/detect/train3/weights/best.pt"
     elixir_timer_model = YOLO(elixir_timer_path)

     runVideo(video_path, hand_model, elixir_timer_model)


if __name__ == "__main__":
     main()









        







