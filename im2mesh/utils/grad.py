import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad


def gradient(inputs, outputs):
    """
    It computes the gradient of the output with respect to the input, but only the last three dimensions
    of the input
    
    :param inputs: the input points to the network
    :param outputs: the output of the network, which is the predicted 3D points
    :return: The gradient of the points with respect to the inputs.
    """
    d_points = torch.ones_like(outputs,
                               requires_grad=False,
                               device=outputs.device)
    points_grad = grad(outputs=outputs,
                       inputs=inputs,
                       grad_outputs=d_points,
                       create_graph=True,
                       retain_graph=True,
                       only_inputs=True)[0][:, -3:]
    return points_grad


def gradient2(inputs, outputs):
    """
    It computes the gradient of the output with respect to the input, but only for the first three
    coordinates of the input
    
    :param inputs: the input points
    :param outputs: the output of the network, which is the predicted points
    :return: The gradient of the points with respect to the inputs.
    """
    d_points = torch.ones_like(outputs,
                               requires_grad=False,
                               device=outputs.device)
    points_grad = grad(outputs=outputs,
                       inputs=inputs,
                       grad_outputs=d_points,
                       create_graph=True,
                       retain_graph=True,
                       only_inputs=True)[0][:, :-3:]
    return points_grad