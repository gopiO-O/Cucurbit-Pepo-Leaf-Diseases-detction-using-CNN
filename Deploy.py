import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('leaf_disease_classifier_model.h5')

class_names = ['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf', 'Mosaic Disease', 'Powdery Mildew']

sample_image_path = r'E:\Projects\Cucurbit Pepo Leaf Diseases\Mosaic_Disease-16-_jpg.rf.d77b314867d08604b4eca0ec7662c353.jpg'

img = image.load_img(sample_image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class]

output_image = cv2.imread(sample_image_path)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output_image, predicted_class_name, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Predicted Image', output_image)
print("Predicted output is: ", predicted_class_name)
cv2.waitKey(0)
cv2.destroyAllWindows()

#output_image_path = 
#cv2.imwrite(output_image_path, output_image)
