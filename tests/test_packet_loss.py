import torch
from scipy import datasets

from src.nn import PacketLoss, MixedLoss

def prepare_input():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face / 255.0
    face = face.permute(0, 3, 1, 2)
    return face


def test_packet_loss_unit_weights():
    face = prepare_input()
    packet_loss = PacketLoss(face, face, wavelet='Haar', level=3, norm_fn='weighted', norm_weights=[1.]*64)
    assert packet_loss.item() == 0.0


def test_packet_loss_without_norm():
    face = prepare_input()
    packet_loss = PacketLoss(face, face, wavelet='Haar', level=3, norm_fn=None, norm_weights=None)
    assert packet_loss.item() == 0.0


def test_packet_loss_log_scale():
    face = prepare_input()
    packet_loss = PacketLoss(face, face, wavelet='Haar', level=3, norm_fn='log', norm_weights=None)
    assert packet_loss.item() == 0.0


def test_mixed_loss_all_combinations():
    face = prepare_input()
    mixed_loss_1 = MixedLoss(face, face, sigma=0.5, wavelet='Haar', level=3, norm_fn=None, norm_weights=None)
    mixed_loss_2 = MixedLoss(face, face, sigma=0.6, wavelet='db3', level=3, norm_fn='log', norm_weights=None)
    mixed_loss_3 = MixedLoss(face, face, sigma=0.8, wavelet='Haar', level=3, norm_fn="weighted", norm_weights=[1.]*64)
    assert mixed_loss_1.item() == 0.0
    assert mixed_loss_2.item() == 0.0
    assert mixed_loss_3.item() == 0.0
