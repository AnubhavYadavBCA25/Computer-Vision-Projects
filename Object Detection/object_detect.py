################################# IMPORTANT NOTE #################################

###### NOTE: I didn't able to upload the yolov3.weights file in this repo. due to it's large size. ##########
###### So, to make this run properly download the weights file from given link and put that file in 'requirements' folder ##########
###### You can download it from the link: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights ##########
###### After downloading the file, put it in 'requirements' folder and run the code. ##########
###### Make sure you follow the above steps to run the code properly. ##########

# Import relevant libraries
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("Object Detection/requirements/yolov3.weights", "Object Detection/requirements/yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

if isinstance(unconnected_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_layers - 1]]

# Load class names (COCO dataset)
with open("Object Detection/requirements/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
img = cv2.imread("Object Detection/images/cat.jpeg")       # IMAGE NEED TO UPLOAD
height, width, channels = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Analyze the results
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        if isinstance(detection, np.ndarray) and detection.shape[0] >= 6:
            scores = detection[5:]  # Extract class scores
            class_id = np.argmax(scores)  # Get the index of the class with the highest score
            confidence = scores[class_id]  # Get the highest score as confidence

            if confidence > 0.5:
                # Object detected, proceed with bounding box extraction
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        else:
            print(f"Invalid detection vector: {detection}")

# Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes on the image
for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0,255,0)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} {round(confidence, 2)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()