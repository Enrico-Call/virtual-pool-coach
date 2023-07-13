from .constants import MODEL_PATH
from .model import IMAGE_TRANSFORM, SmallModel

from pathlib import Path
from .ball_dataset import PoolBallDataset

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

BATCH_SIZE = 16  # 4

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
# BALL_DATA_DIR = PROJ_ROOT / 'data' / 'data-2'
BALL_DATA_TRAIN_DIR = PROJ_ROOT / 'data' / 'data-2-combined'
# BALL_DATA_TEST_DIR = PROJ_ROOT / 'data' / 'data-2-test'
BALL_DATA_TEST_DIR = BALL_DATA_TRAIN_DIR
LOSS_CRITERION = nn.CrossEntropyLoss()


def evaluate(model, dataloader):
    with torch.no_grad():
        total_loss = 0
        correct_count = 0
        total_count = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

            loss = LOSS_CRITERION(outputs, labels)
            total_loss += loss.item()

            # the class with the highest energy is what we choose as prediction
            _, predicted_idxs = torch.max(outputs, 1)
            total_count += labels.size(0)
            correct_count += (predicted_idxs == labels).sum().item()

    assert total_count == len(dataloader.dataset)

    return (total_loss / total_count, correct_count / total_count)


def train(n_epochs=12):
    # Load and normalize

    train_set = PoolBallDataset(BALL_DATA_TRAIN_DIR, IMAGE_TRANSFORM)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_set = PoolBallDataset(BALL_DATA_TEST_DIR, IMAGE_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # net = LargeModel()
    # net = LargeModel.load_as_base(Path(__file__).resolve().parent / 'cifar-model.bin', 10)
    net = SmallModel()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the network
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = LOSS_CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()

            # # print statistics
            # running_loss += loss.item()

        train_loss, train_acc = evaluate(net, train_loader)
        test_loss, test_acc = evaluate(net, test_loader)
        print(
            f'epoch {epoch + 1:3d}\t'
            f'train loss: {train_loss:.3f}\t'
            f'train acc: {train_acc:.3f}\t'
            f'test loss: {test_loss:.3f}\t'
            f'test acc: {test_acc:.3f}\t'
        )

    print('Finished Training')

    # save model
    torch.save(net.state_dict(), MODEL_PATH)


"""
epoch   1	train loss: 0.084	train acc: 0.473	test loss: 0.100	test acc: 0.444	
epoch   2	train loss: 0.073	train acc: 0.554	test loss: 0.103	test acc: 0.315	
epoch   3	train loss: 0.052	train acc: 0.586	test loss: 0.097	test acc: 0.241	
epoch   4	train loss: 0.040	train acc: 0.930	test loss: 0.122	test acc: 0.574	
epoch   5	train loss: 0.030	train acc: 0.941	test loss: 0.083	test acc: 0.796	
epoch   6	train loss: 0.025	train acc: 0.946	test loss: 0.062	test acc: 0.852	
epoch   7	train loss: 0.022	train acc: 0.946	test loss: 0.071	test acc: 0.815	
epoch   8	train loss: 0.021	train acc: 0.946	test loss: 0.073	test acc: 0.815	
epoch   9	train loss: 0.021	train acc: 0.946	test loss: 0.088	test acc: 0.778	
epoch  10	train loss: 0.019	train acc: 0.946	test loss: 0.059	test acc: 0.889	
epoch  11	train loss: 0.018	train acc: 0.946	test loss: 0.051	test acc: 0.889	
epoch  12	train loss: 0.018	train acc: 0.946	test loss: 0.063	test acc: 0.870	
epoch  13	train loss: 0.017	train acc: 0.946	test loss: 0.068	test acc: 0.870	
epoch  14	train loss: 0.017	train acc: 0.946	test loss: 0.062	test acc: 0.870
"""
