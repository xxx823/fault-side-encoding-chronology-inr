import numpy as np
import copy
import utils
import lossf
from mlps import SimpleMLP, ConcatMLP
import torch
import time
import torch.nn as nn

def _safe_unit_np(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def extend_surface_along_normal(points_normed_xyz: np.ndarray, normal_unit_xyz: np.ndarray, delta: float):
    """
    Create three training layers along a supplied normal in normalized space.

    The layers are p- = p - delta*n, p0 = p, and p+ = p + delta*n,
    with scalar labels -1, 0, and +1, respectively.
    """
    n = _safe_unit_np(normal_unit_xyz.astype(np.float32))
    p0 = points_normed_xyz.astype(np.float32)
    p_minus = p0 - delta * n[None, :]
    p_plus  = p0 + delta * n[None, :]

    y0 = -np.ones((len(p0),), dtype=np.float32)
    y1 =  np.zeros((len(p0),), dtype=np.float32)
    y2 =  np.ones((len(p0),), dtype=np.float32)

    P = np.vstack([p_minus, p0, p_plus])
    Y = np.hstack([y0, y1, y2])
    return P, Y

def fault_normal_from_ori_vectors(select_arrow_vectors: np.ndarray, extent: list):
    """
    Convert physical orientation vectors to normalized-space directions and
    estimate a robust global fault normal.

    With u = 2(x - mean) / range, the normalized-space direction is
    proportional to n_phys * (range_x/2, range_y/2, range_z/2).
    """
    dxr = (extent[1] - extent[0]) / 2.0
    dyr = (extent[3] - extent[2]) / 2.0
    dzr = (extent[5] - extent[4]) / 2.0
    scale = np.array([dxr, dyr, dzr], dtype=np.float32)

    v = select_arrow_vectors.astype(np.float32) * scale[None, :]  # -> u-space
    # Normalize each orientation vector before aggregation.
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    # Align vector signs before averaging to avoid cancellation.
    m = vn.mean(axis=0)
    m = _safe_unit_np(m)
    sign = np.sign((vn @ m).reshape(-1, 1))
    sign[sign == 0] = 1
    vn2 = vn * sign

    m2 = vn2.mean(axis=0)
    return _safe_unit_np(m2)









def single_surface_ConcatMLP(interface_points, orientation_points, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta=1, concat=False, epochs=2000, lr=0.001, device=torch.device("cpu")):
    """
    Notes: 1.this function is used to model the single surface, such as fault, unconformity, etc.
           2. use autograd to calculate the orientation gradient
    interface_points: interface points of all kinds of structural interfaces, , which is the original data from .csv files,
                      the format should be a pandas dataframe with columns ['X', 'Y', 'Z']
    orientation_points: orientation points of all kinds of structural interfaces, , which is the original data from .csv files,
                        the format should be a pandas dataframe with columns ['X', 'Y', 'Z', 'dx', 'dy', 'dz']
    extent: the boundary of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus' for better modeling surface
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    """
    # read the data
    interf = interface_points[['X', 'Y', 'Z']].values
    orie = orientation_points[['X','Y','Z','dx','dy','dz']].values
    select_arrow_points = orie[:,:3] 
    select_arrow_vectors = orie[:,3:]
    # normalize the data
    normalized_inter_points = utils.normalize(interf, extent)
    normalized_orien_points = utils.normalize(select_arrow_points, extent)
    # extend the surface, the single surface is extended to three surfaces, the middle one is the original surface
    points, labels = utils.extend_surface(normalized_inter_points)  # labels are the assigned scalar values of the three surface points
    points = np.vstack([points, normalized_orien_points])  # orientation points do not need to be extended to three surfaces
    # convert to torch tensor and send to GPU
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
    mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)
    # the number of interface points and orientation points for dividing the input tensor to send to loss functions
    n_inter = normalized_inter_points.shape[0]
    n_orien = normalized_orien_points.shape[0]
    # the unit direction vectors of the orientation points, used in the loss function as the true orientation
    direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)
    # train the model
    model = ConcatMLP(in_dim=in_dim,
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            n_hidden_layers=n_hidden_layers,
                            activation=activation,
                            beta=beta,
                            concat=concat,
                            ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Initialize variables to track minimum loss and corresponding parameters
    min_loss = float('inf')
    best_params = None
    for epoch in range(epochs):
        y_pred = model(mlp_x_tensor)
        loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)
        loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
        loss = loss_i + 0.1*loss_o     # the weight of the orientation loss is 0.1 is fixed, can set it as a hyperparameter for more flexibility      
        # Check if current loss is lower than minimum loss
        if loss_i < min_loss:
            min_loss = loss_i
            min_loss_i = loss_i
            min_loss_o = loss_o
            # Save the current parameters of the model
            best_params = copy.deepcopy(model.state_dict())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()             
    # query points
    test_x = utils.query_points_exect_bounds(extent, resolution)
    norm_text_x = utils.normalize(test_x, extent)
    test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
    best_model = ConcatMLP(in_dim=in_dim,
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            n_hidden_layers=n_hidden_layers,
                            activation=activation,
                            beta=beta,
                            concat=concat,
                            ).to(device)
    best_model.load_state_dict(best_params)
    with torch.no_grad():
        predictions = best_model(test_x_tensor).cpu().numpy()
    # convert to pyvista mesh
    surface_mesh, grid_mesh_final = utils.predict_to_mesh_single_surface(extent, resolution, predictions)
    print(f'Finish modeling surface | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
    print('------Finish-------')
    
    return surface_mesh


def unconformity_ConcatMLP_neighbor(interface_points, orientation_points, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta_list=[], concat=False, epochs=2000, lr=0.001, delta_orie=1, 
                    device=torch.device("cpu")):
    """
    Notes: 1. use the neighbor points to calculate the orientation gradient, do not use autograd
           2. this function is used to model the unconformity surface, which is an extension of 'single_surface_ConcatMLP', 
           enabling the modeling of multiple unconformity surfaces.
    interface_points: interface points of all kinds of structural interfaces, which is the original data from .csv files
    orientation_points: orientation points of all kinds of structural interfaces, which is the original data from .csv files
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta_list: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    delta_orie: the delta value for calculating the orientation gradient, set as 1 means the gradient is calculated by the difference of 1 cell
    """
    # read the data
    surf_point = interface_points[interface_points['type'] == 'unconformity']
    orie_point = orientation_points[orientation_points['type'] == 'unconformity']
    predic_list = [] # store the multiple predicted unconnformity surfaces
    n = 0  # the index of the beta parameter
    # loop for each unconformity surface
    for name in surf_point['formation'].unique():
        beta = beta_list[n]
        # observation in unconformity
        unconformity_point = surf_point[surf_point['formation'] == name]
        unconformity_orien = orie_point[orie_point['formation'] == name]
        interf = unconformity_point[['X', 'Y', 'Z']].values
        orie = unconformity_orien[['X','Y','Z','dx','dy','dz']].values
        select_arrow_points = orie[:,:3] 
        select_arrow_vectors = orie[:,3:]
        # normalize the data
        normalized_train_data_x = utils.normalize(interf, extent)
        normalized_train_orie_x = utils.normalize(select_arrow_points, extent)
        # extend the surface, this step after normalization to make the surface extension canbe fixed in the same range
        train_data_x, train_data_y = utils.extend_surface(normalized_train_data_x)
        # the additional orientation points for calculating the orientation gradient
        normalized_additional_orie_x6 = utils.delta_cal_orie(normalized_train_orie_x, resolution=resolution, delta=delta_orie)
         
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # concatenate the interface points and orientation points
        train_x_dx = np.vstack((train_data_x, normalized_additional_orie_x6))
        # convert to torch tensor and send to GPU
        x_dx_tensor = torch.tensor(train_x_dx, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(train_data_y, dtype=torch.float32).to(device)
        dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)

        # the number of interface points and orientation points for dividing the input tensor to send to loss functions
        n_inter = normalized_train_data_x.shape[0]
        n_orien = select_arrow_points.shape[0]

        # train the model
        model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Initialize variables to track minimum loss and corresponding parameters
        min_loss = float('inf')
        best_params = None
        for epoch in range(epochs):
            y_pred = model(x_dx_tensor)
            loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], y_tensor)
            # calculate the orientation gradient loss
            xyz_grad = utils.compute_gradient(y_pred[3*n_inter:,:], n_orien, resolution=resolution, delta=delta_orie)
            loss_o = utils.calculate_grad_loss(xyz_grad[:,:], dy_tensor[:,:])

            #loss = loss_i + 0.1*loss_o  # the weight of the orientation loss is 0.1 is fixed
            loss = loss_i + loss_o /(loss_o / loss_i).detach()    # scale the orientation loss to the same range as the interface loss  
            # Check if current loss is lower than minimum loss
            if loss_i < min_loss:
                min_loss = loss_i
                min_loss_i = loss_i
                min_loss_o = loss_o
                # Save the current parameters of the model
                best_params = copy.deepcopy(model.state_dict())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()             
        # query points
        test_x = utils.query_points_exect_bounds(extent, resolution)
        norm_text_x = utils.normalize(test_x, extent)
        test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
        best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
        best_model.load_state_dict(best_params)
        with torch.no_grad():
            predictions = best_model(test_x_tensor).cpu().numpy()
        # convert to pyvista mesh
        unconformity_mesh, grid_mesh_final = utils.predict_to_mesh_single_surface(extent, resolution, predictions)
        print(f'Finish modeling {name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
        predic_list.append(unconformity_mesh)
        n += 1
    print('------Finish-------')
    # output the grid_mesh_final for further visualization, note that the grid_mesh_final only the last one, do not store all the grid_mesh_final
    return predic_list, grid_mesh_final


def unconformity_ConcatMLP(interface_points, orientation_points, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta_list=[], concat=False, epochs=2000, lr=0.001, device=torch.device("cpu")):
    """
    Notes: 1. use the autograd to calculate the orientation gradient
           2. this function is used to model the unconformity surface, which is an extension of 'single_surface_ConcatMLP', 
           enabling the modeling of multiple unconformity surfaces.
    interface_points: interface points of all kinds of structural interfaces, which is the original data from .csv files
    orientation_points: orientation points of all kinds of structural interfaces, which is the original data from .csv files
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta_list: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    """
    # read the data
    surf_point = interface_points[interface_points['type'] == 'unconformity']
    orie_point = orientation_points[orientation_points['type'] == 'unconformity']
    predic_list = [] # store the multiple predicted unconnformity surfaces
    n = 0  # the index of the beta parameter
    # loop for each unconformity surface
    for name in surf_point['formation'].unique():
        beta = beta_list[n]
        # observation in fault
        unconformity_point = surf_point[surf_point['formation'] == name]
        unconformity_orien = orie_point[orie_point['formation'] == name]
        interf = unconformity_point[['X', 'Y', 'Z']].values
        orie = unconformity_orien[['X','Y','Z','dx','dy','dz']].values
        select_arrow_points = orie[:,:3] 
        select_arrow_vectors = orie[:,3:]
        # normalize the data
        normalized_inter_points = utils.normalize(interf, extent)
        normalized_orien_points = utils.normalize(select_arrow_points, extent)
        # extend the surface, the single surface is extended to three surfaces, the middle one is the original surface
        points, labels = utils.extend_surface(normalized_inter_points)
        points = np.vstack([points, normalized_orien_points]) 
        # convert to torch tensor and send to GPU
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
        mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
        mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)
        # train the model 
        n_inter = normalized_inter_points.shape[0]
        n_orien = normalized_orien_points.shape[0]
        direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)
        
        model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Initialize variables to track minimum loss and corresponding parameters
        min_loss = float('inf')
        best_params = None
        for epoch in range(epochs):
            y_pred = model(mlp_x_tensor)
            loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)  #3*n_inter is the number of interface points after extending
            loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
            loss = loss_i + 0.1*loss_o  # the weight of the orientation loss is 0.1 is fixed
            #loss = loss_i + loss_o /(loss_o / loss_i).detach()        
            # Check if current loss is lower than minimum loss
            if loss_i < min_loss:
                min_loss = loss_i
                min_loss_i = loss_i
                min_loss_o = loss_o
                # Save the current parameters of the model
                best_params = copy.deepcopy(model.state_dict())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()             
        # query points
        test_x = utils.query_points_exect_bounds(extent, resolution)
        norm_text_x = utils.normalize(test_x, extent)
        test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
        best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
        best_model.load_state_dict(best_params)
        with torch.no_grad():
            predictions = best_model(test_x_tensor).cpu().numpy()
        # convert to pyvista mesh
        unconformity_mesh, grid_mesh_final = utils.predict_to_mesh_single_surface(extent, resolution, predictions)
        print(f'Finish modeling {name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
        predic_list.append(unconformity_mesh)
        n += 1
    print('------Finish-------')
    # output the grid_mesh_final for further visualization, note that the grid_mesh_final only the last one, do not store all the grid_mesh_final
    return predic_list, grid_mesh_final


def fault_ConcatMLP(interface_points, orientation_points, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta_list=[], concat=False, epochs=2000, lr=0.001, above_below=False, 
                    device=torch.device("cpu"), logger=None):
    """
    Notes: 1. this function is used to model the fault surface, it is divided into two modes by whether using the above and below constraints, 
           2. the above and below constraints are used when the stratigraphic units (points) are very close to the fault surface
           3. use autograd to calculate the orientation gradient
    interface_points: interface points of all kinds of structural interfaces, which is the original data from .csv files
    orientation_points: orientation points of all kinds of structural interfaces, which is the original data from .csv files
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta_list: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    above_below: whether to use the above and below constraints, default is False
    """
    # read the data
    fault_surf_point = interface_points[interface_points['type'] == 'fault']
    fault_orie_point = orientation_points[orientation_points['type'] == 'fault']
    mesh = []
    n = 0
    # whether to use the above and below constraints
    if above_below:
        # loop for each fault surface
        for fault_name in fault_surf_point['formation'].unique():
            beta = beta_list[n]
            # observation in fault
            fault_point = fault_surf_point[fault_surf_point['formation'] == fault_name]
            fault_orien = fault_orie_point[fault_orie_point['formation'] == fault_name]
            interf = fault_point[['X', 'Y', 'Z']].values
            orie = fault_orien[['X','Y','Z','dx','dy','dz']].values
            select_arrow_points = orie[:,:3] 
            select_arrow_vectors = orie[:,3:]
            # observation in above and below of fault surface
            unit_point = interface_points[interface_points['type'] == 'stratigraphic']
            unit_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
            above_point = unit_point[unit_point['ref_'+fault_name] == 'above'][['X', 'Y', 'Z']].values
            below_point = unit_point[unit_point['ref_'+fault_name] == 'below'][['X', 'Y', 'Z']].values
            above_orien = unit_orien[unit_orien['ref_'+fault_name] == 'above'][['X', 'Y', 'Z']].values
            below_orien = unit_orien[unit_orien['ref_'+fault_name] == 'below'][['X', 'Y', 'Z']].values
            above = np.concatenate((above_point, above_orien), axis=0)
            below = np.concatenate((below_point, below_orien), axis=0)
            # # normalize the data
            # normalized_inter_points = utils.normalize(interf, extent)
            # normalized_orien_points = utils.normalize(select_arrow_points, extent)
            # normalized_above_points = utils.normalize(above, extent)
            # normalized_below_points = utils.normalize(below, extent)
            # # extend the surface, the single surface is extended to three surfaces, the middle one is the original surface
            # points, labels = utils.extend_surface(normalized_inter_points)
            # points = np.vstack([points, normalized_above_points, normalized_below_points, normalized_orien_points])
            # # convert to torch tensor and send to GPU
            # #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            # mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            # mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)
            # # the number of interface points, orientation points, above and below for dividing the input tensor to send to loss functions
            # n_inter = normalized_inter_points.shape[0]
            # n_orien = normalized_orien_points.shape[0]
            # n_above = normalized_above_points.shape[0]
            # n_below = normalized_below_points.shape[0]
            # # the unit direction vectors of the orientation points, used in the loss function as the true orientation
            # direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)

            # normalize the data
            normalized_inter_points = utils.normalize(interf, extent)
            normalized_orien_points = utils.normalize(select_arrow_points, extent)
            normalized_above_points = utils.normalize(above, extent)
            normalized_below_points = utils.normalize(below, extent)

            # Build three fault-side training layers along the normalized fault normal.
            n_fault = fault_normal_from_ori_vectors(select_arrow_vectors, extent)
            delta = 2.0 / (min(resolution) - 1)
            points, labels = extend_surface_along_normal(normalized_inter_points, n_fault, delta)

            points = np.vstack([points, normalized_above_points, normalized_below_points, normalized_orien_points])

            # convert to torch tensor and send to GPU
            mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)

            # the number of interface points, orientation points, above and below for dividing the input tensor to send to loss functions
            n_inter = normalized_inter_points.shape[0]
            n_orien = normalized_orien_points.shape[0]
            n_above = normalized_above_points.shape[0]
            n_below = normalized_below_points.shape[0]

            # Transform orientation vectors to normalized coordinates and normalize them.
            dxr = (extent[1] - extent[0]) / 2.0
            dyr = (extent[3] - extent[2]) / 2.0
            dzr = (extent[5] - extent[4]) / 2.0
            scale_t = torch.tensor([dxr, dyr, dzr], dtype=torch.float32, device=device)

            dy_scaled = mlp_dy_tensor * scale_t[None, :]
            direction_vectors_unit = dy_scaled / (torch.norm(dy_scaled, dim=1, keepdim=True) + 1e-12)

            model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Initialize variables to track minimum loss and corresponding parameters
            min_loss = float('inf')
            best_params = None
            for epoch in range(epochs):
                y_pred = model(mlp_x_tensor)
                loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)
                loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
                loss_a = lossf.loss_above(y_pred, n_inter, n_above, device)
                loss_b = lossf.loss_below(y_pred, n_inter, n_above, n_below, device)
                loss = loss_i + 0.1*loss_o + 0.1*loss_a + 0.1*loss_b    
                #loss = loss_i + 0.1*loss_o + 0.1*loss_a + 0.1*loss_b         
                # Check if current loss is lower than minimum loss
                if loss_i < min_loss:
                    min_loss = loss_i
                    min_loss_i = loss_i
                    min_loss_o = loss_o
                    min_loss_ab = loss_a + loss_b
                    # Save the current parameters of the model
                    best_params = copy.deepcopy(model.state_dict())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()             
            # query points
            test_x = utils.query_points(extent, resolution)
            norm_text_x = utils.normalize(test_x, extent)
            test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
            best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            best_model.load_state_dict(best_params)
            with torch.no_grad():
                predictions = best_model(test_x_tensor).cpu().numpy()
            # convert to pyvista mesh
            fault_mesh = utils.predict_to_mesh_fault(extent, resolution, predictions)
            print(f'Finish modeling {fault_name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}, Loss_ab:{min_loss_ab.item()}')
            mesh.append(fault_mesh)
            n += 1
        print('------Finish-------')
    else:
        for fault_name in fault_surf_point['formation'].unique():
            beta = beta_list[n]
            # observation in fault
            fault_point = fault_surf_point[fault_surf_point['formation'] == fault_name]
            fault_orien = fault_orie_point[fault_orie_point['formation'] == fault_name]
            interf = fault_point[['X', 'Y', 'Z']].values
            orie = fault_orien[['X','Y','Z','dx','dy','dz']].values
            select_arrow_points = orie[:,:3] 
            select_arrow_vectors = orie[:,3:]
            # # normalize the data
            # normalized_inter_points = utils.normalize(interf, extent)
            # normalized_orien_points = utils.normalize(select_arrow_points, extent)
            # # extend the surface
            # points, labels = utils.extend_surface(normalized_inter_points)
            # points = np.vstack([points, normalized_orien_points])
            # # convert to torch tensor and send to GPU
            # #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            # mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            # mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)
            #
            # n_inter = normalized_inter_points.shape[0]
            # n_orien = normalized_orien_points.shape[0]
            # direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)

            # normalize the data
            normalized_inter_points = utils.normalize(interf, extent)
            normalized_orien_points = utils.normalize(select_arrow_points, extent)

            # Build three fault-side training layers along the normalized fault normal.
            # Estimate the global normal from the orientation vectors.
            n_fault = fault_normal_from_ori_vectors(select_arrow_vectors, extent)

            # Use approximately one normalized grid spacing for the extension distance.
            delta = 2.0 / (min(resolution) - 1)

            # Generate the three layers along the normal.
            points, labels = extend_surface_along_normal(normalized_inter_points, n_fault, delta)

            # Orientation points are not duplicated along the normal.
            points = np.vstack([points, normalized_orien_points])

            # convert to torch tensor and send to GPU
            mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)

            n_inter = normalized_inter_points.shape[0]
            n_orien = normalized_orien_points.shape[0]

            # Transform orientation vectors to normalized coordinates and normalize them.
            dxr = (extent[1] - extent[0]) / 2.0
            dyr = (extent[3] - extent[2]) / 2.0
            dzr = (extent[5] - extent[4]) / 2.0
            scale_t = torch.tensor([dxr, dyr, dzr], dtype=torch.float32, device=device)

            dy_scaled = mlp_dy_tensor * scale_t[None, :]
            direction_vectors_unit = dy_scaled / (torch.norm(dy_scaled, dim=1, keepdim=True) + 1e-12)

            model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Initialize variables to track minimum loss and corresponding parameters
            min_loss = float('inf')
            best_params = None
            for epoch in range(epochs):
                y_pred = model(mlp_x_tensor)
                loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)
                loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
                loss = loss_i + 0.1*loss_o
                if logger is not None:
                    logger.log(
                        stage_name=f'fault_{fault_name}',
                        epoch=epoch + 1,
                        total_loss=loss.item(),
                        interface_loss=loss_i.item(),
                        orientation_loss=loss_o.item(),
                        above_below_loss=0.0,
                        lr=optimizer.param_groups[0]['lr'],
                    )
                # Check if current loss is lower than minimum loss
                if loss_i < min_loss:
                    min_loss = loss_i
                    min_loss_i = loss_i
                    min_loss_o = loss_o
                    # Save the current parameters of the model
                    best_params = copy.deepcopy(model.state_dict())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()             
            # query points
            test_x = utils.query_points(extent, resolution)
            norm_text_x = utils.normalize(test_x, extent)
            test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
            best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            best_model.load_state_dict(best_params)
            with torch.no_grad():
                predictions = best_model(test_x_tensor).cpu().numpy()
            # convert to pyvista mesh
            fault_mesh = utils.predict_to_mesh_fault(extent, resolution, predictions)
            print(f'Finish modeling {fault_name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
            mesh.append(fault_mesh)
            n += 1
        print('------Finish-------')
    
    return mesh


def fault_ConcatMLP_with_encoding(interface_points, orientation_points, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta_list=[], concat=False, epochs=2000, lr=0.001, above_below=False, 
                    fault_direct='right', movement='down', device=torch.device("cpu")):
    """
    Notes: 1. this function is used to model the fault surface, it is divided into two modes by whether using the above and below constraints, 
           2. the above and below constraints are used when thestratigraphic units (points) are vary close to the fault surface
           3. use autograd to calculate the orientation gradient
           4. directly use the encoding the fault features here
    interface_points: interface points of all kinds of structural interfaces, which is the original data from .csv files
    orientation_points: orientation points of all kinds of structural interfaces, which is the original data from .csv files
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta_list: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    above_below: whether to use the above and below constraints, default is False
    """
    # read the data
    fault_surf_point = interface_points[interface_points['type'] == 'fault']
    fault_orie_point = orientation_points[orientation_points['type'] == 'fault']
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # query points
    test_x = utils.query_points(extent, resolution)
    norm_text_x = utils.normalize(test_x, extent)
    test_x_tensor = torch.tensor(norm_text_x, dtype=torch.float32).to(device)
    mesh = []
    features = []
    features.append(test_x)
    n = 0
    # whether to use the above and below constraints
    if above_below:
        # loop for each fault surface
        for fault_name in fault_surf_point['formation'].unique():
            beta = beta_list[n]
            fault_direction = fault_direct[n]
            move_trend = movement[n]
            # observation in fault
            fault_point = fault_surf_point[fault_surf_point['formation'] == fault_name]
            fault_orien = fault_orie_point[fault_orie_point['formation'] == fault_name]
            interf = fault_point[['X', 'Y', 'Z']].values
            orie = fault_orien[['X','Y','Z','dx','dy','dz']].values
            select_arrow_points = orie[:,:3] 
            select_arrow_vectors = orie[:,3:]
            # observation in above and below of fault surface
            unit_point = interface_points[interface_points['type'] == 'stratigraphic']
            unit_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
            above_point = unit_point[unit_point['ref_'+fault_name] == 'above'][['X', 'Y', 'Z']].values
            below_point = unit_point[unit_point['ref_'+fault_name] == 'below'][['X', 'Y', 'Z']].values
            above_orien = unit_orien[unit_orien['ref_'+fault_name] == 'above'][['X', 'Y', 'Z']].values
            below_orien = unit_orien[unit_orien['ref_'+fault_name] == 'below'][['X', 'Y', 'Z']].values
            above = np.concatenate((above_point, above_orien), axis=0)
            below = np.concatenate((below_point, below_orien), axis=0)
            # normalize the data
            normalized_inter_points = utils.normalize(interf, extent)
            normalized_orien_points = utils.normalize(select_arrow_points, extent)
            normalized_above_points = utils.normalize(above, extent)
            normalized_below_points = utils.normalize(below, extent)
            # extend the surface, the single surface is extended to three surfaces, the middle one is the original surface
            points, labels = utils.extend_surface(normalized_inter_points)
            points = np.vstack([points, normalized_above_points, normalized_below_points, normalized_orien_points])
            # convert to torch tensor and send to GPU
            #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)
            # the number of interface points, orientation points, above and below for dividing the input tensor to send to loss functions
            n_inter = normalized_inter_points.shape[0]
            n_orien = normalized_orien_points.shape[0]
            n_above = normalized_above_points.shape[0]
            n_below = normalized_below_points.shape[0]
            # the unit direction vectors of the orientation points, used in the loss function as the true orientation
            direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)

            model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Initialize variables to track minimum loss and corresponding parameters
            min_loss = float('inf')
            best_params = None
            for epoch in range(epochs):
                y_pred = model(mlp_x_tensor)
                loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)
                loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
                loss_a = lossf.loss_above(y_pred, n_inter, n_above, device)
                loss_b = lossf.loss_below(y_pred, n_inter, n_above, n_below, device)
                loss = loss_i + 0.1*loss_o + 0.1*loss_a + 0.1*loss_b    
                #loss = loss_i + 0.1*loss_o + 0.1*loss_a + 0.1*loss_b         
                # Check if current loss is lower than minimum loss
                if loss_i < min_loss:
                    min_loss = loss_i
                    min_loss_i = loss_i
                    min_loss_o = loss_o
                    min_loss_ab = loss_a + loss_b
                    # Save the current parameters of the model
                    best_params = copy.deepcopy(model.state_dict())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()             
            best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            best_model.load_state_dict(best_params)
            with torch.no_grad():
                predictions = best_model(test_x_tensor).cpu().numpy()
            # convert to pyvista mesh
            fault_mesh = utils.predict_to_mesh_fault(extent, resolution, predictions)
            # assign the fault features
            if fault_direction == 'right':
                flag = 1 if move_trend == 'up' else 0
                encoding_value = np.where(predictions > 0, flag, 1-flag)
            else:
                flag = 0 if move_trend == 'up' else 0
                encoding_value = np.where(predictions > 0, flag, 1-flag)
            print(f'Finish modeling {fault_name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}, Loss_ab:{min_loss_ab.item()}')
            mesh.append(fault_mesh)
            features.append(encoding_value)
            n += 1
        print('------Finish-------')
    else:
        for fault_name in fault_surf_point['formation'].unique():
            beta = beta_list[n]
            fault_direction = fault_direct[n]
            move_trend = movement[n]
            # observation in fault
            fault_point = fault_surf_point[fault_surf_point['formation'] == fault_name]
            fault_orien = fault_orie_point[fault_orie_point['formation'] == fault_name]
            interf = fault_point[['X', 'Y', 'Z']].values
            orie = fault_orien[['X','Y','Z','dx','dy','dz']].values
            select_arrow_points = orie[:,:3] 
            select_arrow_vectors = orie[:,3:]
            # normalize the data
            normalized_inter_points = utils.normalize(interf, extent)
            normalized_orien_points = utils.normalize(select_arrow_points, extent)
            # extend the surface
            points, labels = utils.extend_surface(normalized_inter_points)
            points = np.vstack([points, normalized_orien_points])
            # convert to torch tensor and send to GPU
            #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mlp_x_tensor = torch.tensor(points, dtype=torch.float32).to(device).requires_grad_(True)
            mlp_y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            mlp_dy_tensor = torch.tensor(select_arrow_vectors, dtype=torch.float32).to(device)

            n_inter = normalized_inter_points.shape[0]
            n_orien = normalized_orien_points.shape[0]
            direction_vectors_unit = mlp_dy_tensor / torch.norm(mlp_dy_tensor, dim=1, keepdim=True)

            model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Initialize variables to track minimum loss and corresponding parameters
            min_loss = float('inf')
            best_params = None
            for epoch in range(epochs):
                y_pred = model(mlp_x_tensor)
                loss_i = lossf.loss_intf(y_pred.squeeze()[:3*n_inter], mlp_y_tensor)
                loss_o = lossf.loss_grad(mlp_x_tensor, y_pred, direction_vectors_unit, n_orien)
                loss = loss_i + 0.1*loss_o           
                # Check if current loss is lower than minimum loss
                if loss_i < min_loss:
                    min_loss = loss_i
                    min_loss_i = loss_i
                    min_loss_o = loss_o
                    # Save the current parameters of the model
                    best_params = copy.deepcopy(model.state_dict())
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()             
            best_model = ConcatMLP(in_dim=in_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim,
                              n_hidden_layers=n_hidden_layers,
                              activation=activation,
                              beta=beta,
                              concat=concat,
                              ).to(device)
            best_model.load_state_dict(best_params)
            with torch.no_grad():
                predictions = best_model(test_x_tensor).cpu().numpy()
            # convert to pyvista mesh
            fault_mesh = utils.predict_to_mesh_fault(extent, resolution, predictions)
            # assign the fault features
            if fault_direction == 'right':
                flag = 1 if move_trend == 'up' else 0
                encoding_value = np.where(predictions > 0, flag, 1-flag)
            else:
                flag = 0 if move_trend == 'up' else 0
                encoding_value = np.where(predictions > 0, flag, 1-flag)
            print(f'Finish modeling {fault_name} | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
            mesh.append(fault_mesh)
            features.append(encoding_value)
            n += 1
        print('------Finish-------')
    
    return mesh, features


def stratigraphic_ConcatMLP_neighbor(interface_data, orientation_data, meshgrid_data, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta=1, concat=False, epochs=2000, lr=0.001, delta_orie=1, alpha=0.1, 
                    device=torch.device("cpu")):
    """
    Notes: 1. this function is used to model the stratigraphic surfaces, fault feature encoding for this purpose
           2. use neighbor points of orientation to calculate the orientation gradient
    interface_data: the data comes from the feature encoding results, the list is [label, x, y, z, fault1, fault2, ...], 
                    label value in range [-1, 1], fault1, fault2 are the fault feature encoding results
    orientation_data: the data comes from the feature encoding results, the list is [x, y, z, dx, dy, dz, fault1, fault2, ...],
    meshgrid_data: the meshgrid points for predicting the domain, the format is [x, y, z, fault1, fault2, ...]
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    delta_orie: the delta value for calculating the orientation gradient, set as 1 means the gradient is calculated by the difference of 1 cell
    alpha: the weight of the orientation loss, default is 0.1
    """
    # read the data
    train_data_x = interface_data[:, 1:].astype(np.float)
    train_data_y = interface_data[:, 0]
    train_orie_x = np.delete(orientation_data.copy(), [3,4,5], axis=1).astype(np.float64)
    train_orie_y = orientation_data[:,3:6].astype(np.float64)
    # normalize the data
    normalized_train_data_x = utils.normalize(train_data_x, extent)
    normalized_train_orie_x = utils.normalize(train_orie_x, extent)
    normalized_meshgrid_data = utils.normalize(meshgrid_data, extent)
    # key step: the additional orientation points for calculating the orientation gradient should with the same fault feature as orientation points
    normalized_additional_orie_x6 = utils.delta_cal_orie(normalized_train_orie_x, resolution=resolution, delta=delta_orie)

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # concatenate the interface points and orientation points for training
    train_x_dx = np.vstack((normalized_train_data_x, normalized_additional_orie_x6))
    # convert to torch tensor and send to GPU
    x_dx_tensor = torch.tensor(train_x_dx, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(train_data_y, dtype=torch.float32).to(device)
    dy_tensor = torch.tensor(train_orie_y, dtype=torch.float32).to(device)
    test_x_tensor = torch.tensor(normalized_meshgrid_data, dtype=torch.float32).to(device)
    # the number of interface points and orientation points for dividing the input tensor to send to loss functions
    n_intf = normalized_train_data_x.shape[0]
    n_orie = normalized_train_orie_x.shape[0]
    # train the model
    model = ConcatMLP(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        n_hidden_layers=n_hidden_layers,
                        activation=activation,
                        beta=beta,
                        concat=concat,
                        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  #Adam
    # Initialize variables to track minimum loss and corresponding parameters
    min_loss = float('inf')
    best_params = None

    t1_train = time.time()
    for epoch in range(epochs):
        y_pred = model(x_dx_tensor)  
        # calculate the interface loss
        loss_i = lossf.loss_intf(y_pred[:n_intf,:].squeeze(), y_tensor)
        # calculate the orientation gradient loss
        xyz_grad = utils.compute_gradient(y_pred[n_intf:,:], n_orie, resolution=resolution, delta=delta_orie)
        loss_o = utils.calculate_grad_loss(xyz_grad[:,:], dy_tensor[:,:])
        loss = loss_i + alpha * loss_o
        if loss < min_loss:
            min_loss = loss
            # Save the current parameters of the model
            best_params = copy.deepcopy(model.state_dict())
            min_loss_i = loss_i
            min_loss_o = loss_o
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    t2_train = time.time()
    
    print(f'Training losses | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
    print(f'each epoch training time :  {(t2_train - t1_train) / epochs} seconds')
    # model for inference
    best_model = ConcatMLP(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        n_hidden_layers=n_hidden_layers,
                        activation=activation,
                        beta=beta,
                        concat=concat,
                        ).to(device)
    best_model.load_state_dict(best_params)
    # Predict with the model
    with torch.no_grad():
        t1_inference = time.time()
        predictions = best_model(test_x_tensor).cpu().numpy()
        t2_inference = time.time()
    # convert to pyvista mesh
    # get the iso values for extracting the stratigraphic surfaces
    iso_values = np.unique(interface_data[:, 0])
    stratigraphic_mesh, grid_mesh_final = utils.predict_to_mesh_stratigraphic(extent, resolution, predictions, iso_values)
    print(f'Inference time: {t2_inference - t1_inference} seconds')
    print('------Finish-------')
    
    return stratigraphic_mesh, grid_mesh_final


def stratigraphic_ConcatMLP(interface_data, orientation_data, meshgrid_data, extent, resolution, in_dim, hidden_dim, out_dim,
                    n_hidden_layers, activation='Softplus', beta=1, concat=False, epochs=2000, lr=0.001, alpha=0.1,
                    device=torch.device("cpu"), logger=None):
    """
    Notes: 1. this function is used to model the stratigraphic surfaces, fault feature encoding for this purpose
           2. use 'autograd' to calculate the orientation gradient
    interface_data: the data comes from the feature encoding results, the list is [label, x, y, z, fault1, fault2, ...], 
                    label value in range [-1, 1], fault1, fault2 are the fault feature encoding results
    orientation_data: the data comes from the feature encoding results, the list is [x, y, z, dx, dy, dz, fault1, fault2, ...],
    meshgrid_data: the meshgrid points for predicting the domain, the format is [x, y, z, fault1, fault2, ...]
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    delta_orie: the delta value for calculating the orientation gradient, set as 1 means the gradient is calculated by the difference of 1 cell
    alpha: the weight of the orientation loss, default is 0.1
    """
    # read the data
    train_data_x = interface_data[:, 1:].astype(np.float64)
    train_data_y = interface_data[:, 0]
    train_orie_x = np.delete(orientation_data.copy(), [3,4,5], axis=1).astype(np.float64)
    train_orie_y = orientation_data[:,3:6].astype(np.float64)
    # normalize the data
    normalized_train_data_x = utils.normalize(train_data_x, extent)
    normalized_train_orie_x = utils.normalize(train_orie_x, extent)
    normalized_meshgrid_data = utils.normalize(meshgrid_data, extent)
    # key step: the additional orientation points for calculating the orientation gradient should with the same fault feature as orientation points
    #normalized_additional_orie_x6 = utils.delta_cal_orie(normalized_train_orie_x, resolution=resolution, delta=delta_orie)

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # concatenate the interface points and orientation points for training
    train_x_dx = np.vstack((normalized_train_data_x, normalized_train_orie_x))
    # convert to torch tensor and send to GPU
    x_dx_tensor = torch.tensor(train_x_dx, dtype=torch.float32).to(device).requires_grad_(True)
    y_tensor = torch.tensor(train_data_y, dtype=torch.float32).to(device)
    dy_tensor = torch.tensor(train_orie_y, dtype=torch.float32).to(device)
    test_x_tensor = torch.tensor(normalized_meshgrid_data, dtype=torch.float32).to(device)
    # the number of interface points and orientation points for dividing the input tensor to send to loss functions
    n_intf = normalized_train_data_x.shape[0]
    n_orie = normalized_train_orie_x.shape[0]
    # train the model
    model = ConcatMLP(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        n_hidden_layers=n_hidden_layers,
                        activation=activation,
                        beta=beta,
                        concat=concat,
                        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  #Adam
    # Initialize variables to track minimum loss and corresponding parameters
    min_loss = float('inf')
    best_params = None

    t1_train = time.time()
    for epoch in range(epochs):
        y_pred = model(x_dx_tensor)  
        # calculate the interface loss
        loss_i = lossf.loss_intf_sum(y_pred[:n_intf,:].squeeze(), y_tensor)
        """ criterion = nn.MSELoss(reduction='sum')
        loss_i = criterion(y_pred[:n_intf,:].squeeze(), y_tensor) """
        # calculate the orientation gradient loss
        #xyz_grad = utils.compute_gradient(y_pred[n_intf:,:], n_orie, resolution=resolution, delta=delta_orie)
        #loss_o = utils.calculate_grad_loss(xyz_grad[:,:], dy_tensor[:,:])
        loss_o = lossf.loss_grad_with_fault_features(x_dx_tensor, y_pred, dy_tensor, n_orie)
        loss = loss_i + alpha * loss_o
        if logger is not None:
            logger.log(
                stage_name='stratigraphic_main',
                epoch=epoch + 1,
                total_loss=loss.item(),
                interface_loss=loss_i.item(),
                orientation_loss=loss_o.item(),
                above_below_loss=0.0,
                lr=optimizer.param_groups[0]['lr'],
            )
        if loss < min_loss:
            min_loss = loss
            # Save the current parameters of the model
            best_params = copy.deepcopy(model.state_dict())
            min_loss_i = loss_i
            min_loss_o = loss_o
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    t2_train = time.time()
    
    print(f'Training losses | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
    if logger is not None:
        logger.save_stage_csv('stratigraphic_main')
        logger.plot_stage('stratigraphic_main', smooth_window=31)
    print(f'each epoch training time :  {(t2_train - t1_train) / epochs} seconds')
    # model for inference
    best_model = ConcatMLP(in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        out_dim=out_dim,
                        n_hidden_layers=n_hidden_layers,
                        activation=activation,
                        beta=beta,
                        concat=concat,
                        ).to(device)
    best_model.load_state_dict(best_params)
    # Predict with the model
    with torch.no_grad():
        t1_inference = time.time()
        predictions = best_model(test_x_tensor).cpu().numpy()
        t2_inference = time.time()
    # convert to pyvista mesh
    # get the iso values for extracting the stratigraphic surfaces
    iso_values = np.unique(interface_data[:, 0])
    stratigraphic_mesh, grid_mesh_final = utils.predict_to_mesh_stratigraphic(extent, resolution, predictions, iso_values)
    print(f'Inference time: {t2_inference - t1_inference} seconds')
    print('------Finish-------')
    
    return stratigraphic_mesh, grid_mesh_final
