import cv2
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


from models.unet import UNet
from utils import read_im

in_path = "./test/input"
# fused_path = "./test/fused-ae-alpha1-beta1"
fused_path = "./test/fused-ae-alpha1.5-beta1"

# weights = "./weights/generator_500.pt"
weights = "./weights/autoencoderl1_500_alpha1.5_beta1.pt"



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unet21 = UNet(2, 1).eval()
    unet21.load_state_dict(torch.load(weights, map_location=device))
    print("UNet21 weights loaded successfully!")

    # unet12 = UNet(1, 2).eval()
    # unet12.load_state_dict(torch.load(weights12, map_location=device))
    # print("UNet12 weights loaded successfully!")

    high_paths = []
    low_paths = []
    for filename in os.listdir(in_path):
        if "high" in filename:
            high_paths.append(os.path.join(in_path, filename))
        if "low" in filename:
            low_paths.append(os.path.join(in_path, filename))
    high_paths = sorted(high_paths)
    low_paths = sorted(low_paths)

    data = []
    for low, high in zip(low_paths, high_paths):  # just a check
        if low.split("low")[0] == high.split("high")[0]:
            data.append((low, high))

    # read each image and convert to tensor
    for low, high in data:
        low_im = read_im(low)
        high_im = read_im(high)
        high_im = cv2.resize(high_im, low_im.shape[::-1])

        in_img = np.stack(
            [np.expand_dims(low_im, axis=0), np.expand_dims(high_im, axis=0)], axis=1
        )
        print(in_img.shape)
        intensor = torch.tensor(in_img, dtype=torch.float32, device=device)

        fusedtensor = unet21(intensor).sigmoid()


        fused_im = fusedtensor.detach().numpy()[0, 0, :, :]

        # back_im = backtensor.detach().numpy()
        # low_back, high_back = back_im[0, 0, :, :], back_im[0, 1, :, :]

        save_fused(low, fused_im)
        # save_back(low, high, low_back, high_back)
        # plt.subplot(3, 2, 1)
        # plt.imshow(low_im, cmap="grey")
        # plt.subplot(3, 2, 2)
        # plt.imshow(high_im, cmap="grey")

        # plt.subplot(3, 1, 2)
        # plt.imshow(1-fused_im, cmap="grey")

        # plt.subplot(3, 2, 5)
        # plt.imshow(low_back, cmap="grey")
        # plt.subplot(3, 2, 6)
        # plt.imshow(high_back, cmap="grey")

        # plt.show()
    return None


def save_fused(low_path, fused_im):
    if not os.path.exists(fused_path):
        os.mkdir(fused_path)
    # extract filename
    filename = low_path.split(os.path.sep)[-1]

    # create fusedfilename
    filename = "".join(filename.split("low"))
    # create save path
    save_path = os.path.join(fused_path, filename)
    # save
    fused_im = fused_im * (2**16 - 1)
    cv2.imwrite(save_path, fused_im.astype(np.uint16))
    return None


# def save_back(low_path, high_path, low_back, high_back):
#     if not os.path.exists(back_path):
#         os.mkdir(back_path)
#     # extract filename
#     low_filename = low_path.split(os.path.sep)[-1]
#     high_filename = high_path.split(os.path.sep)[-1]

#     # create save path
#     low_save_path = os.path.join(back_path, low_filename)
#     high_save_path = os.path.join(back_path, high_filename)

#     # save
#     low_back = low_back * (2**16 - 1)
#     high_back = high_back * (2**16 - 1)

#     cv2.imwrite(low_save_path, low_back.astype(np.uint16))
#     cv2.imwrite(high_save_path, high_back.astype(np.uint16))

#     return None


if __name__ == "__main__":
    main()
