# -*- coding: utf-8 -*-
"""
GeoGNN-LSTM
"""
###############################################################################################12-9-3
import warnings
warnings.filterwarnings('ignore')
import os, math
import numpy as np, pandas as pd
import geopandas as gp
import copy

import torch
from torch import nn
import torch.nn.functional as F
from pysal.lib import weights
from scipy.sparse import diags
from shapely.geometry import Point

# enable cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.set_printoptions(suppress=True)
print(device)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def make_sym_matrix_boolean_sum(mx):
    return 1 * np.logical_or(mx, mx.T)

def renormalized_trick_mx(mx):
    return mx + np.eye(mx.shape[0])

def renormalized_trick_laplacian(mx_tilde):
    degree_tilde = np.diag(np.sum(mx_tilde, axis=1))
    D_tilde_inv_sqrt = np.linalg.inv(np.sqrt(degree_tilde))
    return np.dot(D_tilde_inv_sqrt, mx_tilde).dot(D_tilde_inv_sqrt)

# ========== GeoGNN model definition =============
class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=nn.ReLU(), use_bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.use_bias = use_bias
        self.mlp = nn.Linear(in_feats, out_feats, bias=use_bias)
        self.activation = activation

    def forward(self, x, adj):
        agg = torch.mm(adj, x)
        out = self.mlp(agg)
        if self.activation is not None:
            out = self.activation(out)
        return out

class GNN(nn.Module):
    def __init__(self, f_in, n_classes, hidden=[16], dropouts=[0.0]):
        super().__init__()
        layers = []
        in_feats = f_in
        for i, out_feats in enumerate(hidden):
            layers.append(GNNLayer(in_feats, out_feats))
            in_feats = out_feats
        self.layers = nn.ModuleList(layers)
        self.dropouts = dropouts
        self.out_layer = GNNLayer(in_feats, n_classes, activation=nn.Softplus())

    def forward(self, x, adj):
        for layer, d in zip(self.layers, self.dropouts):
            x = layer(x, adj)
            if d > 0:
                x = F.dropout(x, d, training=self.training, inplace=False)
        return self.out_layer(x, adj)

# ========== GeoGNN-LSTM hybrid model (with MC Dropout support) ==============
class GNNLSTMLayer(nn.Module):
    def __init__(self, f_in, gnn_hidden, lstm_hidden, dropout=0.0):
        super().__init__()
        self.gnn_layer = GNNLayer(f_in, gnn_hidden, activation=nn.ReLU())
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
        
    def forward(self, x, adj):
        gnn_out = self.gnn_layer(x, adj)
        if self.dropout > 0:
            gnn_out = F.dropout(gnn_out, self.dropout, training=self.training)
        return gnn_out

class GeoGNN_LSTM(nn.Module):
    """A geospatial deep sequential model combining GeoGNN and LSTM, supporting MC Dropout uncertainty quantification"""
    def __init__(self, f_in, n_classes, gnn_hidden=24, lstm_hidden=32, 
                 num_lstm_layers=2, dropout=0.5, enable_mc_dropout=False):
        super().__init__()
        self.f_in = f_in
        self.gnn_hidden = gnn_hidden
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        self.enable_mc_dropout = enable_mc_dropout  # [Whether to enable MC Dropout]
        
        self.gnn_layer = GNNLSTMLayer(f_in, gnn_hidden, lstm_hidden, dropout)
        
        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # [The output layer includes dropout for MC Dropout]
        self.output_layer = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden // 2, n_classes),
            nn.Softplus()
        )
        
        self.sequence_length = 5  # Reduce sequence length to improve efficiency
        self.train_history_features = None
        ###########################################################################################################################9
        self.n_train_points = 9  # Number of training points
        self.feature_fusion = nn.Linear(lstm_hidden, gnn_hidden)
        
    def reset_history(self):
        self.train_history_features = None
        
    def forward(self, x, adj, update_history=True, train_idx=None):
        """Forward pass with support for MC Dropout mode"""
        batch_size = x.size(0)
        
        # [Core of MC Dropout: keep dropout active even in eval mode]
        if self.enable_mc_dropout:
            gnn_features = self.gnn_layer.gnn_layer(x, adj)
            if self.gnn_layer.dropout > 0:
                gnn_features = F.dropout(gnn_features, self.gnn_layer.dropout, training=True)
        else:
            gnn_features = self.gnn_layer(x, adj)
        
        if train_idx is not None and (self.training or self.enable_mc_dropout):
            train_gnn_features = gnn_features[train_idx]
            current_train_features = train_gnn_features.unsqueeze(0)
            
            if self.train_history_features is None:
                lstm_input = current_train_features.repeat(self.sequence_length, 1, 1)
            else:
                if self.train_history_features.size(0) >= self.sequence_length:
                    lstm_input = torch.cat([
                        self.train_history_features[-(self.sequence_length-1):],
                        current_train_features
                    ], dim=0)
                else:
                    lstm_input = torch.cat([self.train_history_features, current_train_features], dim=0)
                    if lstm_input.size(0) < self.sequence_length:
                        padding_length = self.sequence_length - lstm_input.size(0)
                        padding = current_train_features.repeat(padding_length, 1, 1)
                        lstm_input = torch.cat([padding, lstm_input], dim=0)
            
            lstm_input_batch_first = lstm_input.permute(1, 0, 2)
            lstm_out, (hidden, cell) = self.lstm(lstm_input_batch_first)
            train_lstm_features = lstm_out[:, -1, :]
            enhanced_train_features = self.feature_fusion(train_lstm_features)
            
            enhanced_gnn_features = gnn_features.clone()
            enhanced_gnn_features[train_idx] = enhanced_train_features + gnn_features[train_idx]
            
            if update_history and not self.enable_mc_dropout:
                current_train_features_detached = current_train_features.detach()
                if self.train_history_features is None:
                    self.train_history_features = current_train_features_detached
                else:
                    self.train_history_features = torch.cat([
                        self.train_history_features, 
                        current_train_features_detached
                    ], dim=0)
                    if self.train_history_features.size(0) > self.sequence_length * 2:
                        self.train_history_features = self.train_history_features[-self.sequence_length:]
        else:
            enhanced_gnn_features = gnn_features
        
        # [MC Dropout: force activation of dropout in the output layer]
        if self.enable_mc_dropout:
            for module in self.output_layer:
                if isinstance(module, nn.Dropout):
                    enhanced_gnn_features = module(enhanced_gnn_features)
                elif isinstance(module, nn.Linear):
                    enhanced_gnn_features = module(enhanced_gnn_features)
                elif isinstance(module, nn.ReLU) or isinstance(module, nn.Softplus):
                    enhanced_gnn_features = module(enhanced_gnn_features)
            output = enhanced_gnn_features
        else:
            output = self.output_layer(enhanced_gnn_features)
        
        return output

# ==================== Uncertainty quantification functions ========================

def mc_dropout_predict(model, x_tensor, adj, train_idx, n_samples=50):
    """
    [Method 1: MC Dropout uncertainty quantification]
    
    Returns:
        predictions_mean: Mean predictions [n_nodes, n_features]
        predictions_std: Prediction standard deviation (uncertainty) [n_nodes, n_features]
        all_predictions: All sampled predictions [n_samples, n_nodes, n_features]
    """
    model.eval()
    model.enable_mc_dropout = True
    
    all_predictions = []
    
    print(f"  Running MC Dropout sampling, total {n_samples} times...")
    for i in range(n_samples):
        with torch.no_grad():
            output = model(x_tensor, adj, update_history=False, train_idx=train_idx)
            all_predictions.append(output.cpu().numpy())
        
        if (i + 1) % 10 == 0:
            print(f"    Completed {i+1}/{n_samples} samples")
    
    all_predictions = np.array(all_predictions)  # [n_samples, n_nodes, n_features]
    
    predictions_mean = np.mean(all_predictions, axis=0)
    predictions_std = np.std(all_predictions, axis=0)
    
    model.enable_mc_dropout = False
    
    return predictions_mean, predictions_std, all_predictions


def compute_prediction_variance(output_history, depth_indices=None):
    """
    [Method 3: Predictive variance analysis]
    
    Returns:
        spatial_variance: Spatial variance (variability across depth for each location) [n_nodes] or [n_nodes, n_features]
        temporal_variance: Depth variance (spatial variability within each depth layer) [n_depths] or [n_depths, n_features]
    """
    if depth_indices is None:
        depth_indices = range(len(output_history))
    
    selected_outputs = [output_history[i] for i in depth_indices]
    outputs_array = np.array(selected_outputs)
    
    # Spatial variance: variance across depths for each location (along depth dimension)
    spatial_variance = np.var(outputs_array, axis=0)
    
    # Depth variance: spatial variance within each depth layer (along node dimension)
    temporal_variance = np.var(outputs_array, axis=1)
    
    return spatial_variance, temporal_variance


def compute_information_entropy(predictions_array, n_bins=10):
    """
    [Method 4: Information-entropy uncertainty]
    Compute information entropy based on MC Dropout sampling results
    
    Args:
        predictions_array: Predictions array [n_samples, n_nodes, n_features]
        n_bins: Number of bins for discretization
    
    Returns:
        entropy: Information entropy for each node [n_nodes, n_features]
    """
    n_samples, n_nodes, n_features = predictions_array.shape
    entropy = np.zeros((n_nodes, n_features))
    
    print(f"  Computing information entropy using {n_bins} bins...")
    for node in range(n_nodes):
        for feat in range(n_features):
            values = predictions_array[:, node, feat]
            # Discretization
            hist, bin_edges = np.histogram(values, bins=n_bins, density=True)
            # Compute probabilities
            bin_width = bin_edges[1] - bin_edges[0]
            probabilities = hist * bin_width
            probabilities = probabilities[probabilities > 0]
            # Compute entropy
            if len(probabilities) > 0:
                entropy[node, feat] = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    return entropy


# ==================== Main program ========================
x_vals = np.arange(3, 71 + 1e-6, 1)
y_vals = np.arange(1.23, 91.23 + 1e-6, 1)
z_vals = np.arange(1.5, 9 + 1e-6, 0.03)

records = []
for z in z_vals:
    for y in y_vals:
        for x in x_vals:
            records.append((x, y, z))
df_filtered = pd.DataFrame(records, columns=['x(m)', 'y(m)', 'Depth'])

output_all = []
target_all = []
outputB_all = []
train_all = []

# [Store uncertainty quantification results]
uncertainty_mc_dropout_train = []      # MC Dropout uncertainty (training set)
uncertainty_mc_dropout_test = []       # MC Dropout uncertainty (test set)
# [New: store MC Dropout uncertainty for TestB]
uncertainty_mc_dropout_testB = []      # MC Dropout uncertainty (TestB)

entropy_mc_dropout_train = []          # Information entropy (training set)
entropy_mc_dropout_test = []           # Information entropy (test set)
# [New: store information entropy for TestB]
entropy_mc_dropout_testB = []          # Information entropy (TestB)

predictions_history_train = []         # Training predictions history (for variance analysis)
predictions_history_testA = []         # TestA predictions history
# [New: store TestB prediction history for variance analysis]
predictions_history_testB = []         # TestB predictions history

input_vars = ['x(m)', 'y(m)', 'Depth']
target_var = ['qc']
###########################################################################################################################################################
file_path = 'CPT_Christchurch12_9_3.xlsx'
sheet_names = ['CPT-1', 'CPT-5', 'CPT-8', 'CPT-10', 'CPT-12', 'CPT-14', 'CPT-18', 'CPT-20', 'CPT-24', 'CPT-4', 'CPT-15', 'CPT-21']
train_sheet_names = ['CPT-1', 'CPT-5', 'CPT-8', 'CPT-10', 'CPT-12', 'CPT-14', 'CPT-18', 'CPT-20', 'CPT-24']
n_train = 9
n_testA = 3
n_all_CPT = 12
n_rounds = 251  #[Set to the full number of depth layers]
##########################################################################################################################################################
# ========== Uncertainty quantification configuration ==========
enable_mc_dropout = True      # [Enable MC Dropout]
n_mc_samples = 50             # [Number of MC Dropout samples]
n_entropy_bins = 12          # [Number of bins for information entropy discretization]


n_labels = 1
n_features = len(input_vars)

# [Create a single GNN-LSTM model]
model = GeoGNN_LSTM(
    f_in=n_features,
    n_classes=n_labels,
    gnn_hidden=8*n_features,
    lstm_hidden=16,
    num_lstm_layers=2,
    dropout=0.5,
    enable_mc_dropout=False  # Do not enable during training; enable during inference
).to(device)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=lr)

print(f"\nUncertainty quantification configuration:")
print(f"  - MC Dropout: enabled")
print(f"  - Number of MC samples: {n_mc_samples}")
print(f"  - Predictive variance analysis: enabled")
print(f"  - Information entropy analysis: enabled (bins={n_entropy_bins})")
print(f"  - Total depth layers: {n_rounds}\n")

# ==================== Train and predict layer-by-layer over depth ========================

for rowidx in range(n_rounds):
    print(f"\n{'='*60}")
    print(f"Processing depth layer {rowidx+1}/{n_rounds}, depth: {z_vals[rowidx]:.3f}m")
    print(f"{'='*60}")
    
    # [Read data for the current depth layer]
    rows = []
    for sh in sheet_names:
        df_cpt = pd.read_excel(file_path, sheet_name=sh, nrows=rowidx+1)
        data = df_cpt.iloc[rowidx]
        single_row = [
            data['x(m)'], data['y(m)'], data['Depth'],
            data['qc']
        ]
        rows.append(single_row)
    daily_df = pd.DataFrame(rows, columns=['x(m)', 'y(m)', 'Depth', 'qc'])

    current_depth = daily_df['Depth'].iloc[0]
    tol = 1e-5
    dfB_sub = df_filtered[np.abs(df_filtered['Depth'] - current_depth) < tol].copy()
    dfB_sub['qc'] = 0.
    dfB_sub['geometry'] = dfB_sub.apply(lambda r: Point(r['x(m)'], r['y(m)']), axis=1)

    daily_df['geometry'] = daily_df.apply(lambda row: Point(row['x(m)'], row['y(m)']), axis=1)
    daily_df_big = pd.concat([daily_df, dfB_sub], ignore_index=True)

    db = gp.GeoDataFrame(daily_df_big, geometry='geometry')

    n_B = len(dfB_sub)
    train_idx = np.arange(0, n_train)
    test_idx_a = np.arange(n_train, n_all_CPT)
    test_idx_b = np.arange(n_all_CPT, n_all_CPT + n_B)

    # [Build spatial adjacency matrix]
    knn = weights.KNN.from_dataframe(db, geom_col='geometry', k=2)
    A_sym = make_sym_matrix_boolean_sum(knn.full()[0])
    A_tilde_sym = renormalized_trick_mx(A_sym)
    Laplacian_knn = renormalized_trick_laplacian(A_tilde_sym)
    adj = torch.FloatTensor(Laplacian_knn).to(device)

    x_tensor = torch.FloatTensor(db[input_vars].values).to(device)
    y_tensor = torch.FloatTensor(db[target_var].values).to(device)
    idx_train = torch.LongTensor(train_idx).to(device)
    idx_test_a = torch.LongTensor(test_idx_a).to(device)
    idx_test_b = torch.LongTensor(test_idx_b).to(device)

    # ========== Model training ==========
    print(f"Training model...")
    epochs = 15000
    best_state_dict = None
    lowest_mape = 99999.0
    early_stop_counter = 0
    patience = 3000

    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(x_tensor, adj, update_history=True, train_idx=idx_train)
        loss = F.mse_loss(output[idx_train], y_tensor[idx_train])
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            eval_output = model(x_tensor, adj, update_history=False, train_idx=idx_train)
            global_MAPE = F.l1_loss(eval_output[idx_train], y_tensor[idx_train]) / (y_tensor[idx_train] + 1e-5)
            global_MAPE = global_MAPE.mean().item()
        
        if global_MAPE < lowest_mape - 1e-7:
            lowest_mape = global_MAPE
            best_state_dict = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if i % 3000 == 0 or i == epochs-1:
            print(f"  epoch:{i} --train_loss:{train_loss:.4f} --global_MAPE:{global_MAPE*100:.3f}%")
            
        if early_stop_counter >= patience:
            print(f"  [Early Stop] at epoch {i}")
            break

    # [Load the best model]
    model.load_state_dict(best_state_dict)
    model.eval()

    # ========== MC Dropout uncertainty quantification ==========
    print(f"\nRunning uncertainty quantification:")
    print(f"1. MC Dropout uncertainty...")
    pred_mean, pred_std, all_preds = mc_dropout_predict(
        model, x_tensor, adj, idx_train, n_samples=n_mc_samples
    )
    
    # [Extract predictions and uncertainties for each subset]
    output_train = pred_mean[train_idx]
    output_test_a = pred_mean[test_idx_a]
    output_test_b = pred_mean[test_idx_b]
    
    uncertainty_train = pred_std[train_idx]
    uncertainty_test_a = pred_std[test_idx_a]
    # [New: extract MC Dropout uncertainty for TestB]
    uncertainty_test_b = pred_std[test_idx_b]
    
    target_train = y_tensor[idx_train].cpu().detach().numpy()
    target_test_a = y_tensor[idx_test_a].cpu().detach().numpy()
    
    # [Store MC Dropout uncertainties]
    uncertainty_mc_dropout_train.append(uncertainty_train)
    uncertainty_mc_dropout_test.append(uncertainty_test_a)
    # [New: store MC Dropout uncertainty for TestB]
    uncertainty_mc_dropout_testB.append(uncertainty_test_b)
    
 # ========== Information entropy uncertainty quantification ==========
    print(f"2. Information entropy uncertainty...")
    entropy_all = compute_information_entropy(all_preds, n_bins=n_entropy_bins)
    
    entropy_train = entropy_all[train_idx]
    entropy_test_a = entropy_all[test_idx_a]
    # [New: extract information entropy for TestB]
    entropy_test_b = entropy_all[test_idx_b]
    
    # [Store information entropy]
    entropy_mc_dropout_train.append(entropy_train)
    entropy_mc_dropout_test.append(entropy_test_a)
    # [New: store information entropy for TestB]
    entropy_mc_dropout_testB.append(entropy_test_b)

    # [Store prediction history for variance analysis]
    predictions_history_train.append(output_train)
    predictions_history_testA.append(output_test_a)
    # [New: store TestB prediction history]
    predictions_history_testB.append(output_test_b)

    # [Store training data for each round]
    train_round_data = []
    for i in range(n_train):
        train_round_data.append({
            'CPT_Name': train_sheet_names[i],
            'qc_measured': target_train[i, 0],
            'qc_pred': output_train[i, 0]
        })
    train_all.append(train_round_data)

    output_all.append(output_test_a.flatten())
    target_all.append(target_test_a.flatten())
    outputB_all.append(output_test_b.flatten())

    print(f'Completed depth layer {rowidx+1}/{n_rounds}\n')

# ==================== Compute global uncertainty metrics ========================

print(f"\n{'='*60}")
print("Computing global uncertainty metrics...")
print(f"{'='*60}")

# [3. Predictive variance analysis]
print("\n3. Computing predictive-variance uncertainty...")
predictions_history_train_array = np.array(predictions_history_train)
predictions_history_testA_array = np.array(predictions_history_testA)
# [New: compute predictive variance for TestB]
predictions_history_testB_array = np.array(predictions_history_testB)

spatial_var_train, temporal_var_train = compute_prediction_variance(predictions_history_train_array)
spatial_var_testA, temporal_var_testA = compute_prediction_variance(predictions_history_testA_array)
# [New: compute spatial variance and depth variance for TestB]
spatial_var_testB, temporal_var_testB = compute_prediction_variance(predictions_history_testB_array)

print(f"  Training spatial variance range: [{spatial_var_train.min():.6f}, {spatial_var_train.max():.6f}]")
print(f"  Test spatial variance range: [{spatial_var_testA.min():.6f}, {spatial_var_testA.max():.6f}]")
# [New: output variance info for TestB]
print(f"  TestB spatial variance range: [{spatial_var_testB.min():.6f}, {spatial_var_testB.max():.6f}]")

# ==================== Save results to Excel ========================

print(f"\n{'='*60}")
print("Saving results to Excel...")
print(f"{'='*60}")

# [Convert output/target arrays to numpy]
output_all_np = np.array(output_all)   # shape: [n_rounds, n_testA]
target_all_np = np.array(target_all)
outputB_all_np = np.array(outputB_all)

# [Build Train sheet including all uncertainty information]
train_data_dict = {}
for cpt_name in train_sheet_names:
    train_data_dict[f'{cpt_name}_qc_measured'] = []
    train_data_dict[f'{cpt_name}_qc_pred'] = []
    train_data_dict[f'{cpt_name}_residual'] = []
    train_data_dict[f'{cpt_name}_uncertainty_MC'] = []      # MC Dropout uncertainty
    train_data_dict[f'{cpt_name}_entropy'] = []             # Information entropy

# [Iterate all rounds to fill data]
for round_idx, round_data in enumerate(train_all):
    for cpt_idx, cpt_data in enumerate(round_data):
        cpt_name = cpt_data['CPT_Name']
        qc_measured = cpt_data['qc_measured']
        qc_pred = cpt_data['qc_pred']
        residual = qc_measured - qc_pred
        
        train_data_dict[f'{cpt_name}_qc_measured'].append(qc_measured)
        train_data_dict[f'{cpt_name}_qc_pred'].append(qc_pred)
        train_data_dict[f'{cpt_name}_residual'].append(residual)
        
        # MC Dropout uncertainty
        unc_mc = uncertainty_mc_dropout_train[round_idx][cpt_idx, 0]
        train_data_dict[f'{cpt_name}_uncertainty_MC'].append(unc_mc)
        
        # Information entropy
        entropy_val = entropy_mc_dropout_train[round_idx][cpt_idx, 0]
        train_data_dict[f'{cpt_name}_entropy'].append(entropy_val)

# [Create Train DataFrame]
train_columns = []
for cpt_name in train_sheet_names:
    train_columns.extend([
        f'{cpt_name}_qc_measured',
        f'{cpt_name}_qc_pred',
        f'{cpt_name}_residual',
        f'{cpt_name}_uncertainty_MC',
        f'{cpt_name}_entropy'
    ])

dfTrain = pd.DataFrame(train_data_dict, columns=train_columns)

# ========== Build TestA sheet (New: include confidence interval columns ȳ ± 1.96 σ_MC) ==========
# Note: For each round (each depth layer), put the TestA targets, predictions, and uncertainties into the DataFrame.
# We compute the confidence intervals based on uncertainty_mc_test_np (shape: [n_rounds, n_testA, 1]) and
# output_all_np (shape: [n_rounds, n_testA]).

testA_dict = {
    'target_col1': target_all_np[:, 0],
    'target_col2': target_all_np[:, 1],
    'target_col3': target_all_np[:, 2],
    'output_col1': output_all_np[:, 0],
    'output_col2': output_all_np[:, 1],
    'output_col3': output_all_np[:, 2]
}

# Add TestA uncertainty columns (sigma_MC)
uncertainty_mc_test_np = np.array(uncertainty_mc_dropout_test)  # [n_rounds, n_testA, 1]
entropy_test_np = np.array(entropy_mc_dropout_test)

for i in range(n_testA):
    # σ_MC (standard deviation for this TestA position per round)
    sigma_i = uncertainty_mc_test_np[:, i, 0]
    testA_dict[f'uncertainty_MC_col{i+1}'] = sigma_i
    # Also add information entropy to TestA sheet
    testA_dict[f'entropy_col{i+1}'] = entropy_test_np[:, i, 0]
    
    # ---------- Modification: compute and save two confidence interval methods ----------
    mean_i = output_all_np[:, i]
    
    # [Method 1: Lognormal-distribution confidence interval]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Avoid log(0)
        mean_i_safe = np.maximum(mean_i, 1e-6)
        sigma_i_safe = np.maximum(sigma_i, 1e-6)
        
        # Compute parameters in log space
        # Assume log(qc) ~ N(μ_log, σ_log²)
        sigma_log = np.sqrt(np.log(1 + (sigma_i_safe / mean_i_safe)**2))
        mu_log = np.log(mean_i_safe) - 0.5 * sigma_log**2
        
        # Lognormal confidence interval
        lower_i_lognormal = np.exp(mu_log - 1.96 * sigma_log)
        upper_i_lognormal = np.exp(mu_log + 1.96 * sigma_log)
    
    # [Method 2: Truncated confidence interval (for comparison)]
    lower_i_truncated = np.maximum(mean_i - 1.96 * sigma_i, 0)
    upper_i_truncated = mean_i + 1.96 * sigma_i
    
    # [Save results from both methods]
    # Lognormal method (primary)
    testA_dict[f'output_col{i+1}_ci_lower_lognormal'] = lower_i_lognormal
    testA_dict[f'output_col{i+1}_ci_upper_lognormal'] = upper_i_lognormal
    # Truncated method (comparison)
    testA_dict[f'output_col{i+1}_ci_lower_truncated'] = lower_i_truncated
    testA_dict[f'output_col{i+1}_ci_upper_truncated'] = upper_i_truncated
    # -----------------------------------------------------------------------

dfA = pd.DataFrame(testA_dict)

# [TestB sheet]
testB_cols = [f'outputB_{i + 1}' for i in range(outputB_all_np.shape[1])]
dfB = pd.DataFrame(outputB_all_np, columns=testB_cols)

# [(1) New: build TestB_uncertainty_MC worksheet - in 252*6279 matrix form]
print("Building TestB_uncertainty_MC worksheet...")
uncertainty_testB_np = np.array(uncertainty_mc_dropout_testB)  # [n_rounds, n_testB_nodes, 1]
# Convert to 252*6279 format (remove the last dimension)
uncertainty_testB_matrix = uncertainty_testB_np.squeeze(axis=2)  # [252, 6279]
# Create column names
testB_uncertainty_cols = [f'uncertainty_MC_TestB_{i + 1}' for i in range(uncertainty_testB_matrix.shape[1])]
dfTestB_uncertainty_MC = pd.DataFrame(uncertainty_testB_matrix, columns=testB_uncertainty_cols)

# [(1) New: build TestB_entropy worksheet - in 252*6279 matrix form]
print("Building TestB_entropy worksheet...")
entropy_testB_np = np.array(entropy_mc_dropout_testB)  # [n_rounds, n_testB_nodes, 1]
# Convert to 252*6279 format (remove the last dimension)
entropy_testB_matrix = entropy_testB_np.squeeze(axis=2)  # [252, 6279]
# Create column names
testB_entropy_cols = [f'entropy_TestB_{i + 1}' for i in range(entropy_testB_matrix.shape[1])]
dfTestB_entropy = pd.DataFrame(entropy_testB_matrix, columns=testB_entropy_cols)

# [Uncertainty statistics sheet]
uncertainty_stats_dict = {
    'Depth': z_vals[:n_rounds]
}

# Add spatial variance statistics (variance across depth; one value per CPT)
if spatial_var_train.ndim == 2 and spatial_var_train.shape[1] == 1:
    spatial_var_train_values = spatial_var_train.squeeze()
    for i, cpt_name in enumerate(train_sheet_names):
        uncertainty_stats_dict[f'{cpt_name}_spatial_variance'] = np.full(n_rounds, spatial_var_train_values[i])
elif spatial_var_train.ndim == 1:
    for i, cpt_name in enumerate(train_sheet_names):
        uncertainty_stats_dict[f'{cpt_name}_spatial_variance'] = np.full(n_rounds, spatial_var_train[i])

if spatial_var_testA.ndim == 2 and spatial_var_testA.shape[1] == 1:
    spatial_var_testA_values = spatial_var_testA.squeeze()
    for i in range(n_testA):
        uncertainty_stats_dict[f'TestA_col{i+1}_spatial_variance'] = np.full(n_rounds, spatial_var_testA_values[i])
elif spatial_var_testA.ndim == 1:
    for i in range(n_testA):
        uncertainty_stats_dict[f'TestA_col{i+1}_spatial_variance'] = np.full(n_rounds, spatial_var_testA[i])

# Add depth variance statistics (spatial variance within each depth layer)
if temporal_var_train.ndim == 2 and temporal_var_train.shape[1] == 1:
    uncertainty_stats_dict['temporal_variance_train'] = temporal_var_train.squeeze()
elif temporal_var_train.ndim == 1:
    uncertainty_stats_dict['temporal_variance_train'] = temporal_var_train

if temporal_var_testA.ndim == 2 and temporal_var_testA.shape[1] == 1:
    uncertainty_stats_dict['temporal_variance_testA'] = temporal_var_testA.squeeze()
elif temporal_var_testA.ndim == 1:
    uncertainty_stats_dict['temporal_variance_testA'] = temporal_var_testA

dfUncertainty = pd.DataFrame(uncertainty_stats_dict)

# [(2) New: build TestB_Uncertainty_Stats worksheet]
print("Building TestB_Uncertainty_Stats worksheet...")
testB_uncertainty_stats_dict = {
    'Depth': z_vals[:n_rounds]
}

# [Add TestB spatial variance (fluctuation of each TestB node along 251 layers)]
# spatial_var_testB contains spatial variance value for each TestB node
if spatial_var_testB.ndim == 2 and spatial_var_testB.shape[1] == 1:
    spatial_var_testB_values = spatial_var_testB.squeeze()  # [n_testB_nodes]
    # Each TestB node's spatial variance is constant across all depth layers
    for i in range(len(spatial_var_testB_values)):
        testB_uncertainty_stats_dict[f'TestB_node_{i+1}_spatial_variance'] = np.full(n_rounds, spatial_var_testB_values[i])
elif spatial_var_testB.ndim == 1:
    # Each TestB node's spatial variance is constant across all depth layers
    for i, spatial_var in enumerate(spatial_var_testB):
        testB_uncertainty_stats_dict[f'TestB_node_{i+1}_spatial_variance'] = np.full(n_rounds, spatial_var)

# [Add TestB depth variance (horizontal fluctuation of TestB nodes within each layer)]
# temporal_var_testB contains depth-variance values for each depth layer
if temporal_var_testB.ndim == 2 and temporal_var_testB.shape[1] == 1:
    testB_uncertainty_stats_dict['TestB_temporal_variance'] = temporal_var_testB.squeeze()
elif temporal_var_testB.ndim == 1:
    testB_uncertainty_stats_dict['TestB_temporal_variance'] = temporal_var_testB

dfTestB_Uncertainty_Stats = pd.DataFrame(testB_uncertainty_stats_dict)

# [Save to Excel]
filename = 'GeoGNN_LSTM_Uncertainty_MC_Variance_Entropy_12_9_3.xlsx'
####################################################################################################################################
with pd.ExcelWriter(filename) as writer:
    dfTrain.to_excel(writer, index=False, sheet_name='Train')
    dfA.to_excel(writer, index=False, sheet_name='TestA')
    dfB.to_excel(writer, index=False, sheet_name='TestB')
    dfUncertainty.to_excel(writer, index=False, sheet_name='Uncertainty_Stats')
    # [(1) New: save TestB MC uncertainty and information entropy to new worksheets]
    dfTestB_uncertainty_MC.to_excel(writer, index=False, sheet_name='TestB_uncertainty_MC')
    dfTestB_entropy.to_excel(writer, index=False, sheet_name='TestB_entropy')
    # [(2) New: save TestB variance statistics to a new worksheet]
    dfTestB_Uncertainty_Stats.to_excel(writer, index=False, sheet_name='TestB_Uncertainty_Stats')

print(f'\nResults saved to: {filename}')

# ==================== Print uncertainty statistics summary ========================

print(f"\n{'='*60}")
print("Uncertainty statistics summary")
print(f"{'='*60}")

# [MC Dropout uncertainty statistics]
mc_train_mean = np.mean(uncertainty_mc_dropout_train)
mc_train_std = np.std(uncertainty_mc_dropout_train)
mc_train_min = np.min(uncertainty_mc_dropout_train)
mc_train_max = np.max(uncertainty_mc_dropout_train)

mc_test_mean = np.mean(uncertainty_mc_dropout_test)
mc_test_std = np.std(uncertainty_mc_dropout_test)
mc_test_min = np.min(uncertainty_mc_dropout_test)
mc_test_max = np.max(uncertainty_mc_dropout_test)

# [New: MC Dropout uncertainty statistics for TestB]
mc_testB_mean = np.mean(uncertainty_mc_dropout_testB)
mc_testB_std = np.std(uncertainty_mc_dropout_testB)
mc_testB_min = np.min(uncertainty_mc_dropout_testB)
mc_testB_max = np.max(uncertainty_mc_dropout_testB)

print(f"\n[MC Dropout Uncertainty]")
print(f"  Training set:")
print(f"    Mean={mc_train_mean:.6f}, Std={mc_train_std:.6f}")
print(f"    Range=[{mc_train_min:.6f}, {mc_train_max:.6f}]")
print(f"  Test set:")
print(f"    Mean={mc_test_mean:.6f}, Std={mc_test_std:.6f}")
print(f"    Range=[{mc_test_min:.6f}, {mc_test_max:.6f}]")
# [New: output TestB statistics]
print(f"  TestB:")
print(f"    Mean={mc_testB_mean:.6f}, Std={mc_testB_std:.6f}")
print(f"    Range=[{mc_testB_min:.6f}, {mc_testB_max:.6f}]")

# [Information entropy statistics]
entropy_train_mean = np.mean(entropy_mc_dropout_train)
entropy_train_std = np.std(entropy_mc_dropout_train)
entropy_train_min = np.min(entropy_mc_dropout_train)
entropy_train_max = np.max(entropy_mc_dropout_train)

entropy_test_mean = np.mean(entropy_mc_dropout_test)
entropy_test_std = np.std(entropy_mc_dropout_test)
entropy_test_min = np.min(entropy_mc_dropout_test)
entropy_test_max = np.max(entropy_mc_dropout_test)

# [New: information entropy statistics for TestB]
entropy_testB_mean = np.mean(entropy_mc_dropout_testB)
entropy_testB_std = np.std(entropy_mc_dropout_testB)
entropy_testB_min = np.min(entropy_mc_dropout_testB)
entropy_testB_max = np.max(entropy_mc_dropout_testB)

print(f"\n[Information-Entropy Uncertainty]")
print(f"  Training set:")
print(f"    Mean={entropy_train_mean:.6f}, Std={entropy_train_std:.6f}")
print(f"    Range=[{entropy_train_min:.6f}, {entropy_train_max:.6f}]")
print(f"  Test set:")
print(f"    Mean={entropy_test_mean:.6f}, Std={entropy_test_std:.6f}")
print(f"    Range=[{entropy_test_min:.6f}, {entropy_test_max:.6f}]")
# [New: output TestB statistics]
print(f"  TestB:")
print(f"    Mean={entropy_testB_mean:.6f}, Std={entropy_testB_std:.6f}")
print(f"    Range=[{entropy_testB_min:.6f}, {entropy_testB_max:.6f}]")

# [Spatial variance statistics]
spatial_var_train_mean = np.mean(spatial_var_train)
spatial_var_train_std = np.std(spatial_var_train)
spatial_var_testA_mean = np.mean(spatial_var_testA)
spatial_var_testA_std = np.std(spatial_var_testA)
# [New: TestB spatial variance statistics]
spatial_var_testB_mean = np.mean(spatial_var_testB)
spatial_var_testB_std = np.std(spatial_var_testB)

print(f"\n[Spatial-Variance Uncertainty (Across Depth)]")
print(f"  Training set: Mean={spatial_var_train_mean:.6f}, Std={spatial_var_train_std:.6f}")
print(f"  Test set: Mean={spatial_var_testA_mean:.6f}, Std={spatial_var_testA_std:.6f}")
# [New: output TestB statistics]
print(f"  TestB: Mean={spatial_var_testB_mean:.6f}, Std={spatial_var_testB_std:.6f}")

# [Depth variance statistics]
temporal_var_train_mean = np.mean(temporal_var_train)
temporal_var_train_std = np.std(temporal_var_train)
temporal_var_testA_mean = np.mean(temporal_var_testA)
temporal_var_testA_std = np.std(temporal_var_testA)
# [New: TestB depth variance statistics]
temporal_var_testB_mean = np.mean(temporal_var_testB)
temporal_var_testB_std = np.std(temporal_var_testB)

print(f"\n[Depth-Variance Uncertainty (Within-layer Spatial)]")
print(f"  Training set: Mean={temporal_var_train_mean:.6f}, Std={temporal_var_train_std:.6f}")
print(f"  Test set: Mean={temporal_var_testA_mean:.6f}, Std={temporal_var_testA_std:.6f}")
# [New: output TestB statistics]
print(f"  TestB: Mean={temporal_var_testB_mean:.6f}, Std={temporal_var_testB_std:.6f}")

# [Data dimension info]
print(f"\n[Data Dimension Info]")
print(f"  predictions_history_train length: {len(predictions_history_train)}")
if len(predictions_history_train) > 0:
    print(f"  Single prediction shape: {predictions_history_train[0].shape}")
print(f"  spatial_var_train shape: {spatial_var_train.shape}")
print(f"  spatial_var_testA shape: {spatial_var_testA.shape}")
# [New: output TestB dimension info]
print(f"  spatial_var_testB shape: {spatial_var_testB.shape}")
print(f"  temporal_var_train shape: {temporal_var_train.shape}")
print(f"  temporal_var_testA shape: {temporal_var_testA.shape}")
# [New: output TestB dimension info]
print(f"  temporal_var_testB shape: {temporal_var_testB.shape}")
print(f"  uncertainty_mc_dropout_train length: {len(uncertainty_mc_dropout_train)}")
print(f"  entropy_mc_dropout_train length: {len(entropy_mc_dropout_train)}")
# [New: output TestB dimension info]
print(f"  uncertainty_mc_dropout_testB length: {len(uncertainty_mc_dropout_testB)}")
print(f"  entropy_mc_dropout_testB length: {len(entropy_mc_dropout_testB)}")
print(f"  predictions_history_testB length: {len(predictions_history_testB)}")

# [Uncertainty correlation analysis]
print(f"\n[Correlation Between Uncertainty and Prediction Error]")
all_residuals = []
all_mc_uncertainties = []
all_entropies = []

for round_idx, round_data in enumerate(train_all):
    for cpt_idx, cpt_data in enumerate(round_data):
        residual = abs(cpt_data['qc_measured'] - cpt_data['qc_pred'])
        mc_unc = uncertainty_mc_dropout_train[round_idx][cpt_idx, 0]
        entropy = entropy_mc_dropout_train[round_idx][cpt_idx, 0]
        
        all_residuals.append(residual)
        all_mc_uncertainties.append(mc_unc)
        all_entropies.append(entropy)

# Compute correlation coefficients
corr_mc_residual = np.corrcoef(all_mc_uncertainties, all_residuals)[0, 1]
corr_entropy_residual = np.corrcoef(all_entropies, all_residuals)[0, 1]
corr_mc_entropy = np.corrcoef(all_mc_uncertainties, all_entropies)[0, 1]

print(f"  MC Dropout uncertainty vs absolute residual: {corr_mc_residual:.4f}")
print(f"  Information entropy vs absolute residual: {corr_entropy_residual:.4f}")
print(f"  MC Dropout uncertainty vs information entropy: {corr_mc_entropy:.4f}")

print(f"\n{'='*60}")
print("Uncertainty quantification completed!")
print(f"{'='*60}")

# ==================== Generate uncertainty summary statistics ========================

print("\nGenerating uncertainty summary statistics...")

# [Summary by CPT]
cpt_summary = []
for i, cpt_name in enumerate(train_sheet_names):
    # Extract uncertainties across all depths for this CPT
    mc_uncertainties = [uncertainty_mc_dropout_train[j][i, 0] for j in range(n_rounds)]
    entropies = [entropy_mc_dropout_train[j][i, 0] for j in range(n_rounds)]
    residuals = [abs(train_all[j][i]['qc_measured'] - train_all[j][i]['qc_pred']) for j in range(n_rounds)]
    
    cpt_summary.append({
        'CPT_Name': cpt_name,
        'MC_Uncertainty_Mean': np.mean(mc_uncertainties),
        'MC_Uncertainty_Std': np.std(mc_uncertainties),
        'MC_Uncertainty_Max': np.max(mc_uncertainties),
        'Entropy_Mean': np.mean(entropies),
        'Entropy_Std': np.std(entropies),
        'Entropy_Max': np.max(entropies),
        'Spatial_Variance': spatial_var_train[i, 0] if spatial_var_train.ndim == 2 else spatial_var_train[i],
        'Mean_Absolute_Error': np.mean(residuals),
        'Max_Absolute_Error': np.max(residuals)
    })

dfCPT_Summary = pd.DataFrame(cpt_summary)

# [Summary by depth layer]
depth_summary = []
for j in range(n_rounds):
    mc_uncertainties_depth = uncertainty_mc_dropout_train[j][:, 0]
    entropies_depth = entropy_mc_dropout_train[j][:, 0]
    residuals_depth = [abs(train_all[j][i]['qc_measured'] - train_all[j][i]['qc_pred']) for i in range(n_train)]
    
    depth_summary.append({
        'Depth': z_vals[j],
        'MC_Uncertainty_Mean': np.mean(mc_uncertainties_depth),
        'MC_Uncertainty_Max': np.max(mc_uncertainties_depth),
        'Entropy_Mean': np.mean(entropies_depth),
        'Entropy_Max': np.max(entropies_depth),
        'Temporal_Variance': temporal_var_train[j, 0] if temporal_var_train.ndim == 2 else temporal_var_train[j],
        'Mean_Absolute_Error': np.mean(residuals_depth),
        'Max_Absolute_Error': np.max(residuals_depth)
    })

dfDepth_Summary = pd.DataFrame(depth_summary)

# [Save summary statistics to Excel]
summary_filename = 'GeoGNN_LSTM_Uncertainty_Summary_12_9_3.xlsx'
####################################################################################################################################
with pd.ExcelWriter(summary_filename) as writer:
    dfCPT_Summary.to_excel(writer, index=False, sheet_name='CPT_Summary')
    dfDepth_Summary.to_excel(writer, index=False, sheet_name='Depth_Summary')

print(f"Uncertainty summary statistics saved to: {summary_filename}")
print("\nThe summary includes:")
print("  - CPT_Summary: Uncertainty statistics for each CPT (mean, std, max, etc.)")
print("  - Depth_Summary: Uncertainty statistics for each depth layer")

print(f"\n{'='*60}")
print("All processing completed!")
print(f"{'='*60}")
print(f"\nGenerated files:")
print(f"  1. {filename} - Main results and uncertainty data")
print(f"    Newly added worksheets:")
print(f"      - TestB_uncertainty_MC: MC Dropout uncertainty for TestB (252*6279 matrix)")
print(f"      - TestB_entropy: Information entropy for TestB (252*6279 matrix)")
print(f"      - TestB_Uncertainty_Stats: Variance statistics for TestB")
print(f"  2. {summary_filename} - Uncertainty summary statistics")
print("\nProgram finished.\n")