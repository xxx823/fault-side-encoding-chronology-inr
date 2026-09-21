import torch
import torch.autograd as autograd
import torch.nn as nn

# loss function only for discontinuous stratigraphic 
def loss_intf_sum(y_pred, y_true):
    criterion = nn.MSELoss(reduction='sum')

    return criterion(y_pred, y_true)


# for fault/unconformity
def loss_intf(y_pred, y_true):
    #criterion = nn.MSELoss()
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss(reduction='sum')

    return criterion(y_pred, y_true)


# loss function for the orientation points (wrong)
def loss_grad_wrong(train_x, y_pred, y_true, n_orien):
    """
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    gradients_unit = gradients[-n_orien:,:] / torch.norm(gradients[-n_orien:,:], dim=1, keepdim=True)
    projection_diff = gradients_unit[:, :3] - y_true
    loss_grad = torch.norm(projection_diff, dim=1).mean() # this step is wrong, even the same direction, torch.norm will give a value but not 0

    return loss_grad


# loss function for the orientation points
def loss_grad(train_x, y_pred, y_true, n_orien):
    """
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    grad_norm_pred = torch.norm(gradients[-n_orien:,:], p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', y_true, gradients[-n_orien:,:])
    cosine = grad_inner_product / grad_norm_pred  # orie_tensor using normal orientation
    #loss_grad = torch.sum(1 - cosine)
    loss_grad = torch.mean(1 - cosine)

    return loss_grad


# loss function for the orientation points
def loss_grad_with_fault_features(train_x, y_pred, y_true, n_orien):
    """
    Note: only the first 3 columns of the orientation points are used for the orientation, the rest columns are fault features (locally the same)
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    grad_norm_pred = torch.norm(gradients[-n_orien:,:3], p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', y_true, gradients[-n_orien:,:3])
    cosine = grad_inner_product / grad_norm_pred  # orie_tensor using normal orientation
    loss_grad = torch.sum(1 - cosine)

    return loss_grad


# loss function for the above 
def loss_above(y_pred, n_inter, n_above, device):
    """
    for the points above the interface, the value should be higher than 0 (interface scalar value)
    if the value is lower than 0, then it is a misclassification, minimize the difference with 0
    """
    loss_above = torch.abs(torch.minimum(y_pred[3*n_inter : 3*n_inter+n_above], torch.tensor(0, device=device))).sum()

    return loss_above 


# loss function for the below
def loss_below(y_pred, n_inter, n_above, n_below, device):
    """
    for the points below the interface, the value should be lower than 0 (interface scalar value)
    if the value is higher than 0, then it is a misclassification, minimize the difference with 0
    """
    loss_below = torch.maximum(y_pred[3*n_inter+n_above : 3*n_inter+n_above+n_below], torch.tensor(0, device=device)).sum()

    return loss_below


