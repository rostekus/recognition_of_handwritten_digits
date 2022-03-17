# loading modules
import tkinter as tk
from PIL import Image
import cv2
import io
import os
import numpy as np
from tensorflow.keras import utils
from keras.models import load_model

# window size
WIDTH = 400
HEIGHT = 400

# loading model 
model = load_model('model/my_model.h5')

# creating window app
class Window:
    def __init__(self):
        self.main()

    def main(self):

        # window configuration
        root = tk.Tk()
        root.title('Recognition of handwritten numbers')
        root.geometry('{}x{}'.format(WIDTH,HEIGHT))
        root.resizable(False, False)

        # canvas configuration
        global canvas
        canvas = tk.Canvas(root, width=WIDTH, height=0.7 * HEIGHT)
        canvas.grid(row=0, column=0, columnspan=2)
        canvas.old_coords = None

        # clear buttons 
        clear_button = tk.Button(root, text='CLEAR', font=('Arial', 14), command=lambda x=canvas: x.delete('all'))
        clear_button.grid(row=1, column=0, pady=30)

        # submit button
        submit_button = tk.Button(root, text='CHECK', font=('Arial', 14), command=self.run)
        submit_button.grid(row=1, column=1, pady=30)

        root.bind('<B1-Motion>', self.draw)
        root.bind('<ButtonRelease-1>', self.reset_coords)

        root.mainloop()

    # draw funtion 
    def draw(self, event):
       
        x, y = event.x, event.y
        if canvas.old_coords:
            x1, y1 = canvas.old_coords
            canvas.create_oval(x, y, x1, y1, width=12)
        canvas.old_coords = x, y

    def reset_coords(self, event):
        canvas.old_coords = None

    # save canvas to jpeg image
    def save_drawing(self):

        ps = canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))  # canvas screenshot
        img.save('IMG.jpeg', 'jpeg')

    # tranform saved image for model 
    def image_transormation(self, filename):
        img = Image.open(filename)
        size = (28, 28)
        img = img.resize(size)
        img.save(filename)
        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.bitwise_not(img_array)

        

        img_array_norm  = utils.normalize(img_array, axis=1)

        return img_array_norm
    # using loaded model for prediction
    def predict_number(self):
       
        image = np.array(self.image_transormation('IMG.jpeg'))
        image = image.reshape((1, 28, 28, 1))

        # probability distribution for digits from 0-9
        predicted_values = model.predict(image)  

        # remove comment for more specific data 
        '''
        for number in range(10):
            probability = predicted_values[0][number]
            print('Probability for {}:\t{:.5f}'.format(number, probability))
        '''

        print(f'Predicted: {np.argmax(predicted_values)}') # selecting the biggest probability
        os.remove('IMG.jpeg')  # deletes the drawing after prediction

    def run(self):
        self.save_drawing()
        self.predict_number()

Window()











