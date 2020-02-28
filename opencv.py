import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/wym/Downloads/google.png", 0)

cv2.imshow("google", img)
k = cv2.waitKey(0)
#for 64-bit machine, k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('googlegray.png', img)
    cv2.destroyAllWindows()


img = cv2.imread("/home/wym/Downloads/google.png", 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

#Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. So color
#images will not be displayed correctly in Matplotlib if image is read with Opencv2.
#
#Solution:
#b,g,r = cv2.split(img)
#img = cv2.merge([r, g, b])

#Capture Video from Camera
cap = cv2.VideoCapture(0)
#To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name
#of a video file. Device index is just the number to specify which camera.

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Playing Video from file
cap = cv2.VideoCapture('vtest.avi')
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#Make sure proper versions of ffmpeg or gstreamer is installed. Sometimes, it is a headache to work with Video
#Capture mostly due to wrong installation of ffmpeg/gstreamer.

#Saving a Video
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#or fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

#1.2.3 Drawing Functions in OpenCV
#functions:cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.

#Drawing Line
#lineType;
#– LINE_8 (or omitted) - 8-connected line.
#– LINE_4 - 4-connected line.
#– LINE_AA - antialiased line.

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

#Drawing Rectangle
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

#Drawing Circle
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)

#Drawing Ellipse
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

#Drawing Polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
#cv2.polylines() can be used to draw multiple lines. Just create a list of all the lines you want to draw
#and pass it to the function. All lines will be drawn individually. It is more better and faster way to draw a group of
#lines than calling cv2.line() for each line.

#Adding Text to Images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#1.2.4 Mouse as a Paint-Brush

#Simple Demo

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

#More Advanced Demo

drawing = False
# true if mouse is pressed

mode = True 
# if True, draw rectangle. Press 'm' to toggle to curve

ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()

#Trackbar as the Color Palette

def nothing(x): 
    pass

# Create a black image, a window 
img = np.zeros((300,512,3), np.uint8) 
cv2.namedWindow('image')

# create trackbars for color change 
cv2.createTrackbar('R','image',0,255,nothing) 
cv2.createTrackbar('G','image',0,255,nothing) 
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality 
switch = '0 : OFF \n1 : ON' 
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1): 
    cv2.imshow('image',img) 
    k = cv2.waitKey(1) & 0xFF 
    if k == 27: 
        break
    # get current positions of four trackbars 
    r = cv2.getTrackbarPos('R','image') 
    g = cv2.getTrackbarPos('G','image') 
    b = cv2.getTrackbarPos('B','image') 
    s = cv2.getTrackbarPos(switch,'image')
    if s == 0: 
        img[:] = 0 
    else: 
        img[:] = [b,g,r]

cv2.destroyAllWindows()

#Basic Operations on Images
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\cole\\Desktop\\logo.png")

shape = img.shape

img.item(10, 10, 2)
img.itemset((10, 10, 2), 100)

print(img.size)
print(img.dtype)

#Image ROI
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

#Splitting and Merging Image Channels
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

#or
b = img[:, :, 0]
img[:, :, 2] = 0
# cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing is much more efﬁcient and should be used if possible.

#Making Borders for Images (Padding)
import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

img1 = cv2.imread('C:\\Users\\cole\\Desktop\\logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()

#Arithmetic Operations on Images
#
#Image Addition
#There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation
#while Numpy addition is a modulo operation.
x, y = np.uint8([250]), np.uint8([10])

print(cv2.add(x, y))
print(x+y)

#Image Blending
#𝑑𝑠𝑡 = 𝛼 · 𝑖𝑚𝑔1 + 𝛽 · 𝑖𝑚𝑔2 + 𝛾
#dst(I) = saturate(src1(I) * alpha + src2(I) * beta + gamma)
img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.jpg')
dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Bitwise Operations
# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Performance Measurement and Improvement Techniques
img1 = cv2.imread('messi5.jpg')

e1 = cv2.getTickCount()
for i in range(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print(t)

#Default Optimization in OpenCV
cv2.useOptimized()
%timeit res = cv2.medianBlur(img1,49)

cv2.setUseOptimized(False)
cv2.useOptimized()
%timeit res = cv2.medianBlur(img1,49)

cv2.setUseOptimized(True)

img = cv2.imread('opencv_logo.png')
#Measuring Performance in IPython
x = 5
%timeit y = x**2

%timeit y = x*x

z = np.uint8([5])
%timeit y = z*z

%timeit y = np.square(z)
#Python scalar operations are faster than Numpy scalar operations. So for operations including one or two
#elements, Python scalar is better than Numpy arrays. Numpy takes advantage when size of array is a little bit bigger.

%timeit z = cv2.countNonZero(img)

%timeit z = np.count_nonzero(img)
#Normally, OpenCV functions are faster than Numpy functions. So for same operation, OpenCV functions are
#preferred. But, there can be exceptions, especially when Numpy works with views instead of copies.

#There are several techniques and coding methods to exploit maximum performance of Python and Numpy. Only
#relevant ones are noted here and links are given to important sources. The main thing to be noted here is that, first try
#to implement the algorithm in a simple manner. Once it is working, profile it, find the bottlenecks and optimize them.
#
#1. Avoid using loops in Python as far as possible, especially double/triple loops etc. They are inherently slow.
#2. Vectorize the algorithm/code to the maximum possible extent because Numpy and OpenCV are optimized for
# vector operations.
#3. Exploit the cache coherence.
#4. Never make copies of array unless it is needed. Try to use views instead. Array copying is a costly operation.
#
#Even after doing all these operations, if your code is still slow, or use of large loops are inevitable, use additional
#libraries like Cython to make it faster.


#Image Processing in OpenCV
#Changing Colorspaces

import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

#Object Tracking
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


#How to find HSV values to track?
green = np.uint8([[[0,255,0]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print(hsv_green)
#Now you take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound respectively. Apart from this
#method, you can use any image editing tools like GIMP or any online converters to find these values, but don’t forget
#to adjust the HSV ranges.

#Image Thresholding
#Simple Thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.png',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


#Adaptive Thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dave.jpg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

#Otsu’s Binarization
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisy2.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)', 'Original Noisy Image','Histogram',"Otsu's Thresholding", 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()


#How Otsu’s Binarization Works?
img = cv2.imread('noisy2.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
        
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(thresh,ret)

#Geometric Transformations of Images