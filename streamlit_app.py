import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import face_recognition
import os

from PIL import Image,ImageEnhance
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

#--------------------------------------------- face detection with multiple features ----------------------

@st.cache
def load_image(img):
	im = Image.open(img)
	return im


face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('frecog/haarcascade_smile.xml')

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img,faces


def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img

def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon


def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny

#------------------------------------- END ------------------


#------------------------------------ objection dtection with image with YOLOv3 (func) ---------------------

def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Original Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # YOLO ALGORITHM
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))


    # LOAD THE IMAGE
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # OBJECT DETECTED
                #Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)
    print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = str.upper((classes[class_ids[i]]))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,8)
            items.append(label)



    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")
    plt.figure(figsize = (18,18))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))

#----------------------------------------- YOLO (END Func) ------------------------------------------

# CONSTANTS-------------------------------- Face Detection with Image(function) -------------------
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

# convert image from opened file to np.array


def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

# convert opencv BRG to regular RGB mode


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))
#-------------------------------------Face Detection with Image(END) -----------------------------------
def main():
    activities = ["Face Detection with Multiple Features", "Image Classification", "Objection Dectection with Image", "Face Detection with Image", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)
    st.write()

    if choice == 'Face Detection with Image':

        st.set_option("deprecation.showfileUploaderEncoding", False)

        # title area
        st.title("Face Detection with Image:boy:")

        # displays a file uploader widget and return to BytesIO
        image_byte = st.file_uploader(
            label="Select a picture contains faces:", type=['jpg', 'png']
        )
        # detect faces in the loaded image
        max_faces = 0
        rois = []  # region of interests (arrays of face areas)
        if image_byte is not None:
            image_array = byte_to_array(image_byte)
            face_locations = face_recognition.face_locations(image_array)
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                # save face region of interest to list
                rois.append(image_array[top:bottom, left:right].copy())

                # Draw a box around the face and lable it
                cv2.rectangle(image_array, (left, top),
                              (right, bottom), COLOR_DARK, 2)
                cv2.rectangle(
                    image_array, (left, bottom + 35),
                    (right, bottom), COLOR_DARK, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    image_array, f"#{idx}", (left + 5, bottom + 25),
                    font, .55, COLOR_WHITE, 1
                )

            st.image(BGR_to_RGB(image_array), width=720)
            max_faces = len(face_locations)

        if max_faces > 0:
            # select interested face in picture
            face_idx = st.selectbox("Select face#", range(max_faces))
            roi = rois[face_idx]
            st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

            # initial database for known faces
            DB = init_data()
            face_encodings = DB[COLS_ENCODE].values
            dataframe = DB[COLS_INFO]

            # compare roi to known faces, show distances and similarities
            face_to_compare = face_recognition.face_encodings(roi)[0]
            dataframe['distance'] = face_recognition.face_distance(
                face_encodings, face_to_compare
            )
            dataframe['similarity'] = dataframe.distance.apply(
                lambda distance: f"{face_distance_to_conf(distance):0.2%}"
            )
            st.dataframe(
                dataframe.sort_values("distance").iloc[:5]
                    .set_index('name')
            )

            # add roi to known database
            if st.checkbox('add it to knonwn faces'):
                face_name = st.text_input('Name:', '')
                face_des = st.text_input('Desciption:', '')
                if st.button('add'):
                    encoding = face_to_compare.tolist()
                    DB.loc[len(DB)] = [face_name, face_des] + encoding
                    DB.to_csv(PATH_DATA, index=False)
        else:
            st.write('No human face detected.')


    elif choice=='Image Classification':
        st.title("Image Classification :sunflower:")
        st.sidebar.subheader("Input")
        models_list = ["VGG16", "VGG19", "Inception", "Xception", "ResNet"]
        network = st.sidebar.selectbox("Select the Model", models_list)

        MODELS = {
            "VGG16": VGG16,
            "VGG19": VGG19,
            "Inception": InceptionV3,
            "Xception": Xception,  # TensorFlow ONLY
            "ResNet": ResNet50,
        }

        uploaded_file = st.sidebar.file_uploader(
            "Choose an image to classify", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            bytes_data = uploaded_file.read()

            # initialize the input image shape (224x224 pixels) along with
            # the pre-processing function (this might need to be changed
            # based on which model we use to classify our image)
            inputShape = (224, 224)
            preprocess = imagenet_utils.preprocess_input
            # if we are using the InceptionV3 or Xception networks, then we
            # need to set the input shape to (299x299) [rather than (224x224)]
            # and use a different image pre-processing function
            if network in ("Inception", "Xception"):
                inputShape = (299, 299)
                preprocess = preprocess_input

            Network = MODELS[network]
            model = Network(weights="imagenet")

            # load the input image using PIL image utilities while ensuring
            # the image is resized to `inputShape`, the required input dimensions
            # for the ImageNet pre-trained network
            image = Image.open(BytesIO(bytes_data))
            image = image.convert("RGB")
            image = image.resize(inputShape)
            image = img_to_array(image)
            # our input image is now represented as a NumPy array of shape
            # (inputShape[0], inputShape[1], 3) however we need to expand the
            # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
            # so we can pass it through the network
            image = np.expand_dims(image, axis=0)
            # pre-process the image using the appropriate function based on the
            # model that has been loaded (i.e., mean subtraction, scaling, etc.)
            image = preprocess(image)

            preds = model.predict(image)
            predictions = imagenet_utils.decode_predictions(preds)
            imagenetID, label, prob = predictions[0][0]

            st.image(bytes_data, caption=[f"{label} {prob * 100:.2f}"])
            st.subheader(f"Top Predictions from {network}")
            st.dataframe(
                pd.DataFrame(
                    predictions[0], columns=["Network", "Classification", "Confidence"]
                )
            )

    elif choice == 'Objection Dectection with Image':
        st.title('Objection Dectection with Image using YOLOv3:camera:')
        choices = st.radio("", ("Show Demo", "Browse an Image"))

        if choices == "Browse an Image":
            st.set_option('deprecation.showfileUploaderEncoding', False)
            image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

            if image_file is not None:
                our_image = Image.open(image_file)
                detect_objects(our_image)

        elif choices == "Show Demo":
            our_image = Image.open("images/person.jpg")
            detect_objects(our_image)


    elif choice == 'Face Detection with Multiple Features':
        st.subheader("Face Detection with Multiple Features:octocat:")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            # st.image(our_image,width=300)

            enhance_type = st.sidebar.radio("Enhance Type",
                                            ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == 'Gray-Scale':
                new_img = np.array(our_image.convert('RGB'))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # st.write(new_img)
                st.image(gray)
            elif enhance_type == 'Contrast':
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            elif enhance_type == 'Brightness':
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            elif enhance_type == 'Blurring':
                new_img = np.array(our_image.convert('RGB'))
                blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
                img = cv2.cvtColor(new_img, 1)
                blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(blur_img)
            elif enhance_type == 'Original':

                st.image(our_image, width=300)
            else:
                st.image(our_image, width=300)

        # Face Detection
        task = ["Faces", "Smiles", "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):

            if feature_choice == 'Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)

                st.success("Found {} faces".format(len(result_faces)))
            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img)
                # st.success("Found {} faces".format(len(result_img)))


            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img)
                # st.success("Found {} faces".format(len(result_img)))

            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img)

            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny)


    elif choice == 'About':
        st.header('About:speech_balloon:')
        st.subheader("Multi Task Web App from Deep Learning:thumbsup:")
        st.markdown("Built with Streamlit by Mukhtar Khan:sunglasses::v:")

        st.success("World(:earth_asia:) is itself:shit: Big Data:computer:")









if __name__ == "__main__":
    main()
