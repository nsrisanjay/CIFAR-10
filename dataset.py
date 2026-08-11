import numpy as np
import torch
from normalisation import getDatasetMeanAndSD

trainingFiles = [
               "./dataset/data_batch_1.bin",
               "./dataset/data_batch_2.bin",
               "./dataset/data_batch_3.bin",
               "./dataset/data_batch_4.bin",
               "./dataset/data_batch_5.bin"
           ]

testingFiles = ["./dataset/test_batch.bin"]

mean,SD = getDatasetMeanAndSD(trainingFiles,32,32,50000,3)

class CIFAR10Dataset:
    def __init__(self,train:bool,imageWidth:int,imageHeight:int,channels:int):
        if train is True:
           self.totalImages = 50000
           self.filePaths = trainingFiles

        else:
            self.totalImages = 10000
            self.filePaths = testingFiles

        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.channels = channels
        # get mean and standard deviation
        self.mean,self.SD = mean,SD
        # normalise pixel values /255
        self.mean = torch.from_numpy(self.mean/255.0).to(torch.float32)
        self.SD = torch.from_numpy(self.SD/255.0).to(torch.float32)


    def __len__(self):
        return self.totalImages

    def __getitem__(self, index):
        # first find which file
        fileNumber = index//10000
        filePath = self.filePaths[fileNumber]
        with open(filePath,"rb") as file:
            file.seek(3073*(index%10000))
            imageByteBuffer = file.read(3073)
            file.close()

        # now extract labels and RGB channels
        label = imageByteBuffer[0]
        redChannelPixels = torch.frombuffer(imageByteBuffer[1:1025],dtype = torch.uint8).reshape((self.imageWidth,self.imageHeight))
        greenChannelPixels = torch.frombuffer(imageByteBuffer[1025:2049],dtype = torch.uint8).reshape((self.imageWidth,self.imageHeight))
        blueChannelPixels = torch.frombuffer(imageByteBuffer[2049:3073],dtype = torch.uint8).reshape((self.imageWidth,self.imageHeight))

        finalImageArray = torch.empty((self.channels,self.imageWidth,self.imageHeight),dtype = torch.uint8)
        finalImageArray[0] = redChannelPixels
        finalImageArray[1] = greenChannelPixels
        finalImageArray[2] = blueChannelPixels

        # convert the dtype into float32
        finalImageArray = finalImageArray.to(torch.float32)

        del redChannelPixels,greenChannelPixels,blueChannelPixels

        '''
        why normalise each channel again??
        We normalize each RGB channel separately because each channel has a different distribution
        , meaning its pixels can have different means and standard deviations.
        These differences affect the scale of the inputs, which in turn affects
        the gradients calculated from the loss function. Since the optimizer uses
        those gradients to decide how much to update each weight, poorly scaled inputs
        can produce poorly scaled updates, making the path toward the minimum less efficient.
        By centering and scaling each channel using its own mean and standard deviation, we make the
        inputs more comparable in scale, which generally gives the optimizer 
        a better-conditioned problem and makes training easier.
        '''

        # normalisation
        finalImageArray = finalImageArray/255.0
        for i in range(self.channels):
            finalImageArray[i] = (finalImageArray[i] - self.mean[i])/self.SD[i]

        return (finalImageArray,label)





