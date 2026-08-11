'''
This version of the code where we read data and convert everything to numpy arrays and then feed these
to the training pipeline involves putting entire datset into memory which is not a good habit, since in memory
processing may be limited and may not be supported for larger datasets exceeding available memory.


so we created another file dataset.py which processes and keeps one image at a time in memory
'''


# read all the labels and store in map
import numpy as np

LABELS_PATH = "./dataset/batches.meta.txt"
NUMBER_OF_IMAGES = 60000
CHANNELS = 3
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

with open(LABELS_PATH,"r") as file:
    labels = file.read()
    labels = labels.split('\n')
    labels.pop()
    labels.pop()
    # print(labels)
    file.close()


# EXTRACT IMAGES
labelsArray = np.empty((NUMBER_OF_IMAGES,),dtype = np.int8)
imagesArray = np.empty((NUMBER_OF_IMAGES,CHANNELS,IMAGE_WIDTH,IMAGE_HEIGHT),dtype = np.int64)

imageIndex = 0

for i in range(1,6):
    with open(f"./dataset/data_batch_{i}.bin","rb") as file:
        while True:
            dataBuffer = file.read(3073)
            labelBuffer = dataBuffer[0]
            labelsArray[imageIndex] = labelBuffer
            redChannelBuffer = dataBuffer[1:1025]
            greenChannelBuffer = dataBuffer[1025:2049]
            blueChannelBuffer = dataBuffer[2049:3073]

            redChannelPixels = np.frombuffer(redChannelBuffer,dtype=np.uint8)
            greenChannelPixels = np.frombuffer(greenChannelBuffer,dtype=np.uint8)
            blueChannelPixels = np.frombuffer(blueChannelBuffer,dtype=np.uint8)

            redChannelPixels = np.reshape((32,32))
            greenChannelPixels = np.reshape((32,32))
            blueChannelPixels = np.reshape((32,32))

            greenChannelPixels

            imagesArray[imageIndex] = np.asarray([redChannelPixels],[greenChannelPixels],[blueChannelPixels])



    
