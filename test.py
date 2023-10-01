from src.data.data import image_generator
import os
import matplotlib.pyplot as plt

x = "/datasets/seg_test"
classes = os.listdir(f"{x}/seg_test")
gen = image_generator(x, classes, (150, 150, 3),0, 15)


for i in gen:
    plt.imshow(i)
    plt.show()
