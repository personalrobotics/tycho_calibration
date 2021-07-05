import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation as scipyR


def dh_transformation(alpha: torch.Tensor,a: torch.Tensor,_theta,d: torch.Tensor,theta_offset: torch.Tensor):
    theta = _theta + theta_offset
    # Given the DH link construct the transformation matrix
    rot = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta)*torch.cos(alpha), torch.cos(theta)*torch.cos(alpha), -torch.sin(alpha)],
                    [torch.sin(theta)*torch.sin(alpha), torch.cos(theta)*torch.sin(alpha), torch.cos(alpha)]])

    trans = torch.tensor([a,-d*torch.sin(alpha),torch.cos(alpha)*d]).reshape(3,1)
    last_row = torch.tensor([[0, 0, 0, 1]])
    m = torch.cat((torch.cat((rot, trans), 1),last_row), 0)
    return m


def fk_transformation(fk_tensor: torch.Tensor, joint_positions) -> torch.Tensor:
    # Given a list of FKparams, shape N by 3, return transformation
    ee = torch.eye(4)
    for (alpha, a, d, offset), theta in zip(fk_tensor, joint_positions):
        ee @= dh_transformation(alpha, a, theta, d, offset)
    return ee


def quat_shift_to_transformation(params):
    # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
    qx,qy,qz,qw,x,y,z = params
    rot = scipyR.from_quat((qx, qy, qz, qw)).as_matrix()
    trans = np.array([x,y,z]).reshape(3,1)
    last_row = np.array([[0, 0, 0, 1]])
    return np.vstack((np.hstack((rot, trans)),last_row))


class Net(nn.Module):
    # fk_params should be 6x4 matrix, where the i-th row gives the DH params for the i-th joint (for first 6 joints)
    def __init__(self, fk_params: np.ndarray):
        super().__init__()
        self.fk_params = torch.tensor(fk_params, requires_grad=True)
        self.layers = nn.Sequential(
            nn.Linear(7, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 7),
            nn.Tanh()
        )


    # this does NOT give the EE position. Instead, it gives the position of the 6th segment.
    def forward(self, joints):
        correction = self.layers.forward(joints)
        wrist_transform = fk_transformation(self.fk_params, joints + correction)
        return wrist_transform[:-1, -1] # x,y,z stored in the upper three rows of the last column


def train(model, data, loss, optimizer):
    pass


def main():
    pass


if __name__ == "__main__":
    main()