import numpy as np

def getDatasetMeanAndSD(filePaths,imageWidth,imageHeight,totalImages,channels):
    pixelsOffset = channels*imageWidth*imageHeight + 1
    # RGBSum = [0.0,0.0,0.0]
    # RGBSqauredSum = [0.0,0.0,0.0]

    RGBSum = np.zeros((3,),dtype=np.float64)
    RGBSqauredSum = np.zeros((3,),dtype=np.float64)

    for file in filePaths:
        with open(file,"rb") as f:
            while True:
                byteBuffer = f.read(pixelsOffset)
                if not byteBuffer:
                    break
                redPixelsSum = np.frombuffer(byteBuffer[1:1025],dtype=np.uint8).astype(np.int64).sum()
                greenPixelSum = np.frombuffer(byteBuffer[1025:2049],dtype=np.uint8).astype(np.int64).sum()
                bluePixelSum = np.frombuffer(byteBuffer[2049:3073],dtype=np.uint8).astype(np.int64).sum()

                redPixelSqauredSum = (np.frombuffer(byteBuffer[1:1025],dtype=np.uint8).astype(np.int64) ** 2).sum()
                greenPixelSquaredSum = (np.frombuffer(byteBuffer[1025:2049],dtype=np.uint8).astype(np.int64) ** 2).sum()
                bluePixelSquaredSum = (np.frombuffer(byteBuffer[2049:3073],dtype=np.uint8).astype(np.int64) ** 2).sum()

                RGBSqauredSum[0] += redPixelSqauredSum
                RGBSqauredSum[1] += greenPixelSquaredSum
                RGBSqauredSum[2] += bluePixelSquaredSum

                RGBSum[0] += redPixelsSum
                RGBSum[1] += greenPixelSum
                RGBSum[2] += bluePixelSum
        f.close()

    for i in range(0,3):
        RGBSum[i] = RGBSum[i]/(totalImages*imageHeight*imageWidth)

    for i in range(0,3):
        RGBSqauredSum[i] = ((RGBSqauredSum[i]/(totalImages*imageHeight*imageWidth)) - RGBSum[i] ** 2) ** 0.5
        
    return (RGBSum,RGBSqauredSum)

