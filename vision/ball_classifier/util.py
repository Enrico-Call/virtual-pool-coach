import numpy as np
import torch
import torchvision
from matplotlib import pyplot


def show_img_batch(batch: torch.Tensor):
    img = torchvision.utils.make_grid(batch)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    pyplot.imshow(np.transpose(npimg, (1, 2, 0)))
    pyplot.show()



# def run():
#     net = LargeModel.load(MODEL_PATH)
#
#     # TODO randomize and split dataset (or one big batch)
#     ds = PoolBallDataset(BALL_DATA_DIR, transform=IMAGE_TRANSFORM)
#
#     img, label = ds[0]
#     img = img.unsqueeze(dim=0)
#     outputs = net(img)
#
#     # the class with the highest energy is what we choose as prediction
#     # _, pred_class_idxs = torch.max(outputs, 1)
#     _, predicted_idxs = torch.max(outputs.data, 1)
#     label_index = int(predicted_idxs[0])
#
#     print(predicted_idxs)
