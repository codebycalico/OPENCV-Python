# import the required modules 
import cv2 
import matplotlib.pyplot as plt 
# DeepFace requires tensorflow to be installed
from deepface import DeepFace 
  
# read image 
img = cv2.imread('img.jpg') 
  
# call imshow() using plt object 
plt.imshow(img[:,:,::-1]) 
  
# display that image 
plt.show() 
  
# storing the result 
result = DeepFace.analyze(img,actions=['emotion']) 
  
# print result 
print(result) 