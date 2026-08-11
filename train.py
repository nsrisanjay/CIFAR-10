import os
import torch
from torch.utils.data import DataLoader

from dataset import CIFAR10Dataset
from model import CNNCifar


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CHANNELS = 3

LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 5


def main():

    train_dataset = CIFAR10Dataset(
        train=True,
        imageWidth=IMAGE_WIDTH,
        imageHeight=IMAGE_HEIGHT,
        channels=CHANNELS
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_dataset = CIFAR10Dataset(
        train=False,
        imageWidth=IMAGE_WIDTH,
        imageHeight=IMAGE_HEIGHT,
        channels=CHANNELS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = CNNCifar()

    lossFunction = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        lossPerEpoch = 0.0
        correctPredictions = 0
        totalPredictions = 0

        for images, labels in train_loader:

            predictedLabels = model(images)

            loss = lossFunction(
                predictedLabels,
                labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossPerEpoch += loss.item()

            predictions = predictedLabels.argmax(dim=1)

            correctPredictions += (
                predictions == labels
            ).sum().item()

            totalPredictions += labels.size(0)

        averageLoss = lossPerEpoch / len(train_loader)

        trainAccuracy = (
            correctPredictions / totalPredictions
        ) * 100

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {averageLoss:.4f} "
            f"Train Accuracy: {trainAccuracy:.2f}%"
        )

    os.makedirs("savedModel", exist_ok=True)

    torch.save(
        model.state_dict(),
        "savedModel/model1.pt"
    )

    print("\nModel saved to savedModel/model1.pt")

    model.eval()

    correctPredictions = 0
    totalPredictions = 0

    with torch.no_grad():

        for images, labels in test_loader:

            predictedLabels = model(images)

            predictions = predictedLabels.argmax(dim=1)

            correctPredictions += (
                predictions == labels
            ).sum().item()

            totalPredictions += labels.size(0)

    testAccuracy = (
        correctPredictions / totalPredictions
    ) * 100

    print(
        f"Test Accuracy: {testAccuracy:.2f}%"
    )


if __name__ == "__main__":
    main()



# from dataset import CIFAR10Dataset
# from torch.utils.data import DataLoader
# from model import CNNCifar
# import torch

# IMAGE_WDITH = 32
# IMAGE_HEIGHT = 32
# CHANNELS = 3
# LEARNING_RATE = 0.001
# EPOCHS = 10

# def main():

#     dataset = CIFAR10Dataset(True,IMAGE_WDITH,IMAGE_HEIGHT,CHANNELS)

#     dataLoader = DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0)

#     model = CNNCifar()

#     lossFunction = torch.nn.CrossEntropyLoss()

#     optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)

#     for epoch in range(EPOCHS):
#         lossPerEpoch = 0
#         for (image,label) in dataLoader:
#             predictedLabel = model(image)
#             loss = lossFunction(predictedLabel,label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lossPerEpoch += loss.item()
#         print("average loss : ", lossPerEpoch/len(dataLoader))


# if __name__ == "__main__":
#     main()