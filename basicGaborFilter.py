import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ksize = 50
sigma = 3
theta = 3*np.pi/4
lamda = 1*np.pi/4
gamma = 2
phi = 0

kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

img = cv2.imread(r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg')
# img = cv2.resize(img, (200, 200))                                            #---> Resize image to 200x200
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

kernel_resized = cv2.resize(kernel, (400, 400))

cv2.imshow('image', img)
cv2.imshow('filtered image', fimg)
cv2.imshow('kernel', kernel_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

