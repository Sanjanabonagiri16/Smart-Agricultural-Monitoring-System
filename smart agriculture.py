import pandas as pd
data = pd.read_csv('sensor_data.csv')
import matplotlib.pyplot as plt
plt.plot(data['timestamp'], data['temperature'])
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.show()
import cv2
from tensorflow.keras.models import load_model
image = cv2.imread('plant_image_1.jpg')
image_resized = cv2.resize(image, (128, 128)) / 255.0
model = load_model('plant_disease_model.h1')
prediction = model.predict(image_resized.reshape(1, 128, 128, 3))
def check_conditions(data):
    if data['temperature'] > 30:
        print("Alert: High temperature!")
    if data['soil_moisture'] < 200:
        print("Alert: Low soil moisture!")

