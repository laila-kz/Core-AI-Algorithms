#Lire image sous OpenCV :
# import cv2
# img1 = cv2.imread('cameraman.jpeg',-1)
# img2 = cv2.imread('cameraman.jpeg',0)
# img3 = cv2.imread('cameraman.tiff')
# print("format de l'image 1 : ", img1.shape)
# print("height : ", img1.shape[0])
# print("width : ", img1.shape[1])
# print("channels : ", img1.shape[2])
# print("taille de l'image 1 : ", img1.size)



import cv2
img = cv2.imread('coins.jpeg',-1)
cv2.imshow('image', img)
cv2.waitKey(60)
cv2.destroyAllWindows()
