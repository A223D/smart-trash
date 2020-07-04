import numpy as np
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
labels=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = load_model('final_model.h5')
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print('Original Dimensions : ',frame.shape)
    scale_percent = 150 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    #rows, cols=cap.shape()

    # Our operations on the frame come here
    #frame = cv2.flip(frame, flipCode = -1)

    cv2.imwrite('pic.jpg', frame)
    img = image.load_img('./pic.jpg', target_size=(300, 300))
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images=np.vstack([x])
    results = model.predict(images)
    idx=np.where(results[0]==np.amax(results[0]))
    print(labels[idx[0][0]])
    cv2.putText(frame, labels[idx[0][0]],
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    lineType)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
