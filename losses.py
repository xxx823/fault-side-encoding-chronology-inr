import torch
import torch.autograd as autograd
import torch.nn as nn

# loss function only for discontinuous stratigraphic 
def loss_intf_sum(y_pred, y_true):
    criterion = nn.MSELoss(reduction='sum')

    return criterion(y_pred, y_true)


def loss_intf_mean(y_pred, y_true):
    """Return the mean squared interface-position error.

    The manuscript defines the interface loss as an average over the
    interface samples.  Keeping this as a separate helper preserves the
    legacy ``loss_intf_sum`` API while making the paper runner explicit.
    """

    criterion = nn.MSELoss(reduction='mean')
    return criterion(y_pred, y_true)


# for fault/unconformity
def loss_intf(y_pred, y_true):
    #criterion = nn.MSELoss()
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss(reduction='sum')

    return criterion(y_pred, y_true)


# loss function for the orientation points
def loss_grad(train_x, y_pred, y_true, n_orien):
    """
    Direction loss for the fault-surface stage in normalized coordinates.

    ``y_true`` is the observed physical normal transformed by the inverse
    coordinate Jacobian and normalized before this function is called.

    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    grad_norm_pred = torch.norm(gradients[-n_orien:,:], p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', y_true, gradients[-n_orien:,:])
    cosine = grad_inner_product / (grad_norm_pred + 1e-12)
    loss_grad = torch.mean(1 - cosine)

    return loss_grad


# loss function for the orientation points
def normalized_to_physical_gradient(gradient_normalized, coordinate_ranges):
    """Convert gradients from normalized coordinates to physical coordinates.

    For ``u_i = 2 * (x_i - center_i) / range_i``, the chain rule gives
    ``grad_x = diag(2 / range_i) @ grad_u``.
    """
    ranges = torch.as_tensor(
        coordinate_ranges,
        dtype=gradient_normalized.dtype,
        device=gradient_normalized.device,
    ).reshape(-1)
    if ranges.numel() != 3 or torch.any(ranges <= 0):
        raise ValueError("coordinate_ranges must contain three positive physical ranges")
    return gradient_normalized[..., :3] * (2.0 / ranges)


def loss_grad_with_fault_features(
    train_x, y_pred, y_true, n_orien, coordinate_ranges
):
    """
    Note: only the first 3 columns of the orientation points are used for the orientation, the rest columns are fault features (locally the same)
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(
        outputs=y_pred,
        inputs=train_x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    physical_gradient = normalized_to_physical_gradient(
        gradients[-n_orien:, :3], coordinate_ranges
    )
    epsilon = torch.finfo(physical_gradient.dtype).eps
    predicted_unit = physical_gradient / torch.linalg.vector_norm(
        physical_gradient, dim=1, keepdim=True
    ).clamp_min(epsilon)
    observed_unit = y_true / torch.linalg.vector_norm(
        y_true, dim=1, keepdim=True
    ).clamp_min(epsilon)
    cosine = torch.sum(predicted_unit * observed_unit, dim=1)
    return torch.mean(1.0 - cosine)


def loss_eikonal_normalized(train_x, y_pred):
    """Apply a unit-gradient regularizer in normalized spatial coordinates.

    Fault-feature channels are held fixed; only the first three input channels
    contribute to the spatial gradient.
    """
    gradients = autograd.grad(
        outputs=y_pred,
        inputs=train_x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    gradient_norm = torch.linalg.vector_norm(gradients[:, :3], dim=1)
    return torch.mean((gradient_norm - 1.0) ** 2)


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

