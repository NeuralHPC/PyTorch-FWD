import numpy as np
import torch
from PIL import Image
from scipy import datasets

from scripts.fid.fid import calculate_frechet_distance
from scripts.fid.inception import InceptionV3


def compute_activation(img_tensor, model):
    op1 = model(img_tensor)[0]
    onp1 = op1.squeeze(3).squeeze(2).cpu().numpy()
    return onp1


if __name__ == "__main__":
    face = datasets.face()
    face = Image.fromarray(face).resize((299, 299))
    face = np.expand_dims(face, 0).astype(np.float32) / 255.0

    face2 = np.round(face, 2)
    face3 = np.round(face, 3)

    prepare = lambda tensor: torch.permute(torch.tensor(tensor), [0, 3, 1, 2])

    face = prepare(face)
    face2 = prepare(face2)
    face3 = prepare(face3)

    bidx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([bidx])
    model = model  # .to('cuda:0')
    model.eval()

    with torch.no_grad():
        act = model(face)[0].squeeze([-2, -1]).numpy()
        act2 = model(face2)[0].squeeze([-2, -1]).numpy()
        act3 = model(face3)[0].squeeze([-2, -1]).numpy()

    def compute_fid(activation1, activation2):
        m1, s1 = np.mean(activation1, axis=0), np.cov(activation1, rowvar=False)
        m2, s2 = np.mean(activation2, axis=0), np.cov(activation2, rowvar=False)
        return calculate_frechet_distance(m1, s1, m2, s2)

    fid2 = compute_fid(act, act2)
    fid3 = compute_fid(act, act3)

    pass
