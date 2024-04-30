import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer, RobustScaler

def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    batch_size = timestamps.shape[0]
    print("constants ", constants.shape)
    print("timestamps ", timestamps.shape)

    timestamps = torch.squeeze(timestamps)
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)

class QuantileTRF():
    def __init__(self):
        self.quantile_trfs_out = None
        self.quantile_trfs_in = None

    def quantile_transform_in(self, passed_data):
        return quantile_transform_trf(passed_data, self.quantile_trfs_in)

    def quantile_transform_out(self, passed_data):
        return quantile_transform_trf(passed_data, self.quantile_trfs_out)

    def quantile_inv_transform_out(self, passed_data):
        return quantile_inv_transform_trf(passed_data, self.quantile_trfs_out)



class NormalizeData():
    def __init__(self):
        self.input_indices = [0, 1, 2, 3]
        #self.output_indices = [0, 1, 2]
        self.input_mean = [0, 0, 0, 0]
        #self.output_mean = [0, 0, 0, 0]
        self.input_std = [1, 1, 1, 1]
        #self.output_std = [1,1,1, 1]
        #self.output_quantiles = []

    def normalize_input(self, data):
        data = (data - self.input_mean) / self.input_std
        return torch.squeeze(data)

    def normalize_output(self, data):
        raise NotImplementedError
        # output_data = torch.empty((data.shape))
        # for i in range(data.shape[0]):
        #     if i in self.output_indices:
        #         if hasattr(self, 'output_quantiles') and len(self.output_quantiles) > 0:
        #             output_data[i][...] =(data[i] - self.output_quantiles[1, i]) / (self.output_quantiles[2, i] - self.output_quantiles[0, i])
        #         else:
        #             output_data[i][...] = (data[i] - self.output_mean[i]) /self.output_std[i]
        #     else:
        #         output_data[i][...] = data[i]
        # return output_data


def log_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log(data[0])

        # flatten_data = data[1].flatten()
        # positive_data_indices = flatten_data >= 1e-15
        # negative_data_indices = flatten_data < 1e-15
        #
        # preprocessed_negative_k_xy = -torch.log(np.abs(flatten_data[negative_data_indices]))
        # preprocessed_positive_k_xy = torch.log(flatten_data[positive_data_indices])
        #
        # flatten_data[positive_data_indices] = preprocessed_positive_k_xy
        # flatten_data[negative_data_indices] = preprocessed_negative_k_xy

        #print("data[1].shape ", data[1].shape)
        #
        # output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[1][...] = torch.log(data[1] + 0.03)#torch.abs(torch.min(data[1])))
        output_data[2][...] = torch.log(data[2])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data

def log10_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log10(data[0])

        flatten_data = data[1].flatten()
        print("log10 flatten data ", flatten_data)
        positive_data_indices = flatten_data >= 1e-15
        negative_data_indices = flatten_data < 1e-15

        preprocessed_negative_k_xy = -torch.log10(np.abs(flatten_data[negative_data_indices]))
        preprocessed_positive_k_xy = torch.log10(flatten_data[positive_data_indices])



        flatten_data[positive_data_indices] = preprocessed_positive_k_xy
        flatten_data[negative_data_indices] = preprocessed_negative_k_xy

        print("log10 final flatten data ", flatten_data)

        output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[2][...] = torch.log10(data[2])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


def init_norm(data):
    bulk_features_avg, input, output, cross_section_flag = data
    #avg_k = torch.mean(input)

    # print("bulk features avg ", bulk_features_avg)
    # print("output ", output)

    if cross_section_flag:
        input[:3, :] /= bulk_features_avg
    else:
        input /= bulk_features_avg
    output /= bulk_features_avg

    #print("output ", output)

    return input, output


def arcsinh_data(data):
    output_data = torch.empty((data.shape))
    #print("data.shape ", data.shape)
    if data.shape[0] == 3:
        output_data[0][...] = data[0]
        output_data[1][...] = torch.arcsinh(data[1])
        print("data[1] ", data[1])
        print("torch.arcsinh(data[1]) ", torch.arcsinh(data[1]))
        output_data[2][...] = data[2]
    # elif data.shape[0] == 4:
    #     output_data[0][...] = torch.log(data[0])
    #     output_data[1][...] = data[1]
    #     output_data[2][...] = torch.log(data[2])
    #     output_data[3][...] = data[3]
    # elif data.shape[0] < 3:
    #     for i in range(data.shape[0]):
    #         output_data[i][...] = torch.log(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data

def log_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log(data[2])
    elif data.shape[0] == 4:
        output_data[0][...] = torch.log(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log(data[2])
        output_data[3][...] = data[3]
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.log(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data


def log10_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.log10(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log10(data[2])
    elif data.shape[0] == 4:
        output_data[0][...] = torch.log10(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.log10(data[2])
        output_data[3][...] = data[3]
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.log10(data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")

    return output_data

def quantile_transform_fit(data, indices=[], transform_type=None):
    transform_obj = []
    for i in range(data.shape[0]):
        if i in indices:
            if transform_type == "RobustScaler":
                transformer = RobustScaler()
            else:
                transformer = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution="normal")
            if torch.is_tensor(data[i]):
                transform_obj.append(transformer.fit(data[i].reshape(-1, 1).numpy()))
            else:
                transform_obj.append(transformer.fit(data[i].reshape(-1, 1)))
        else:
            transform_obj.append(None)
    return transform_obj


def quantile_transform_trf(data, quantile_trfs):
    trf_data = torch.empty((data.shape))
    for i in range(data.shape[0]):
        if quantile_trfs[i] is not None:
            transformed_data = quantile_trfs[i].transform(data[i].reshape(-1, 1).numpy())
            trf_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
        else:
            trf_data[i][...] = data[i]
    return trf_data

def quantile_inv_transform_trf(data, quantile_trfs):
    trf_data = torch.empty((data.shape))
    for i in range(data.shape[0]):
        if quantile_trfs[i] is not None:
            transformed_data = quantile_trfs[i].inverse_transform(data[i].reshape(-1, 1).numpy())
            trf_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
        else:
            trf_data[i][...] = data[i]
    return trf_data

# def quantile_transform_offdiagonal_fit(data, transform_type=None):
#     transform_obj = []
#     for i in range(data.shape[0]):
#         if i == 1:
#             if transform_type == "RobustScaler":
#                 transformer = RobustScaler()
#             else:
#                 transformer = QuantileTransformer(n_quantiles=10000, random_state=0, output_distribution="normal")
#             if torch.is_tensor(data[i]):
#                 transform_obj.append(transformer.fit(data[i].reshape(-1, 1).numpy()))
#             else:
#                 transform_obj.append(transformer.fit(data[i].reshape(-1, 1)))
#     return transform_obj
#
#
# def quantile_transform_offdiagonal_trf(data, quantile_trfs):
#     return_data = torch.empty((data.shape))
#     for i in range(data.shape[0]):
#         if i == 1:
#             #print("data ", data[i])
#             transformed_data = quantile_trfs[0].transform(data[i].reshape(-1, 1).numpy())
#             return_data[i][...] = torch.from_numpy(np.reshape(transformed_data, data[i].shape))
#         else:
#             return_data[i][...] = data[i]
#     return return_data


def exp_data(data):
    print("data shape ", data.shape)
    print("data ", data)

    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.exp(data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.exp(data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.exp(data[i])
    elif len(data.shape) == 4:
        if data.shape[1] == 3:
            data = np.transpose(data, (1, 2, 3, 0))
            output_data = np.transpose(output_data, (1, 2, 3, 0))
            print("data[:][0] ", data[:][0])
            output_data[0][...] = torch.exp(data[0][...])
            output_data[1][...] = data[1][...]
            output_data[2][...] = torch.exp(data[2][...])
            output_data = np.transpose(output_data, (3, 0, 1, 2))
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
    print("output data ", output_data)
    return output_data

def power_10_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.pow(10, data[0])
        output_data[1][...] = data[1]
        output_data[2][...] = torch.pow(10, data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.pow(10, data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
    return output_data


def power_10_all_data(data):
    output_data = torch.empty((data.shape))
    if data.shape[0] == 3:
        output_data[0][...] = torch.pow(10, data[0])
        #output_data[0][...] = torch.exp(data[0])

        flatten_data = data[1].flatten()
        print("flatten data ", flatten_data)

        positive_data_indices = flatten_data >= 1e-15
        negative_data_indices = flatten_data < 1e-15

        print("flatten_data[negative_data_indices]", flatten_data[negative_data_indices])
        print("flatten_data[positive_data_indices]", flatten_data[positive_data_indices])

        #preprocessed_positive_k_xy = -torch.pow(torch.from_numpy(10), flatten_data[negative_data_indices])
        #preprocessed_negative_k_xy = torch.pow(torch.from_numpy(10), -flatten_data[positive_data_indices])

        preprocessed_positive_k_xy = - 10 ** flatten_data[negative_data_indices]
        preprocessed_negative_k_xy = 10 ** -flatten_data[positive_data_indices]

        flatten_data[negative_data_indices] = preprocessed_positive_k_xy
        flatten_data[positive_data_indices] = preprocessed_negative_k_xy

        print("final flatten data ", flatten_data)

        output_data[1][...] = np.reshape(flatten_data, data[1].shape)
        output_data[2][...] = torch.pow(10, data[2])
    elif data.shape[0] < 3:
        for i in range(data.shape[0]):
            output_data[i][...] = torch.pow(10, data[i])
    else:
        raise NotImplementedError("Log transformation implemented for 2D case only")
    return output_data


def get_mean_std(data_loader, output_iqr=[]):
    running_sum = torch.zeros((1, 1, 4))
    running_sum_squares = torch.zeros((1, 1, 4))
    num_values = 0
    for node_features in data_loader:
        print("node features shape ", node_features.shape)
        running_sum += torch.sum(node_features, dim=(0, 1), keepdim=True)
        running_sum_squares += torch.sum(node_features ** 2, dim=(0, 1), keepdim=True)
        num_values += (node_features.shape[0] * node_features.shape[1])  # add batch size * num vertices

    mean = running_sum / num_values
    std = (running_sum_squares / num_values - (mean ** 2)) ** 0.5

    # final_features = (node_features - mean) / std
    #
    # f_features_mean = torch.mean(final_features, dim=(0, 1))
    # f_features_std = torch.std(final_features, dim=(0, 1))
    #
    # print("f features mean ", f_features_mean)
    # print("f features std ", f_features_std)

    return mean, std


def reshape_to_tensors(tn_array, dim=2):
    tn = np.eye(dim)
    tn[np.triu_indices(dim)] = tn_array

    diagonal_values = np.diag(tn)
    symmetric_tn = tn + tn.T
    np.fill_diagonal(symmetric_tn, diagonal_values)
    return symmetric_tn


def get_eigendecomp(flatten_values, dim=2):
    tensor = reshape_to_tensors(np.squeeze(flatten_values), dim=dim)

    return np.linalg.eigh(tensor)


def check_shapes(n_conv_layers, kernel_size, stride, pool_size, pool_stride, pool_indices, input_size=256):
    #n_layers = 0

    for i in range(n_conv_layers):
        # print("input size ", input_size)
        # print("kernel size ", kernel_size)

        if input_size < kernel_size:
            return -1, input_size

        input_size = int(((input_size - kernel_size) / stride)) + 1

        if pool_indices is not None:
            if i in list(pool_indices.keys()):
                if pool_size > 0 and pool_stride > 0:
                    input_size = int(((input_size - pool_size) / pool_stride)) + 1

    return 0, input_size


def get_mse_nrmse_r2(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    squared_err_k = []
    std_tar_k = []
    r2_k = []
    mse_k = []
    rmse_k = []
    nrmse_k = []

    for i in range(targets_arr.shape[1]):
        targets = np.squeeze(targets_arr[:, i, ...])
        predictions = np.squeeze(predictions_arr[:, i, ...])
        squared_err_k.append((targets - predictions) ** 2)
        std_tar_k.append(np.std(targets))
        r2_k.append(1 - (np.sum(squared_err_k[i]) /
                         np.sum((targets - np.mean(targets)) ** 2)))
        mse_k.append(np.mean(squared_err_k[i]))
        rmse_k.append(np.sqrt(mse_k[i]))
        nrmse_k.append(rmse_k[i] / std_tar_k[i])

    return mse_k, rmse_k, nrmse_k, r2_k


def get_mse_nrmse_r2_eigh(targets, predictions):
    targets_arr = np.array(targets)
    predictions_arr = np.array(predictions)

    # print("targets arr shape", targets_arr.shape)
    # print("predictions arr shape ", predictions_arr.shape)

    y_pred_resized = np.empty((predictions_arr.shape[0], 2, 2))
    y_pred_resized[:, 0, 0] = predictions_arr[:, 0]
    y_pred_resized[:, 0, 1] = predictions_arr[:, 1]
    y_pred_resized[:, 1, 0] = predictions_arr[:, 1]
    y_pred_resized[:, 1, 1] = predictions_arr[:, 2]

    y_true_resized = np.empty((targets_arr.shape[0], 2, 2))
    y_true_resized[:, 0, 0] = targets_arr[:, 0]
    y_true_resized[:, 0, 1] = targets_arr[:, 1]
    y_true_resized[:, 1, 0] = targets_arr[:, 1]
    y_true_resized[:, 1, 1] = targets_arr[:, 2]

    # pred_evals = self._calc_evals(y_pred_resized)

    ####
    ## Eigenvalues of predicted tensors
    ####
    eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
              np.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                         4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                              y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

    eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
              np.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                         4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                              y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

    pred_evals = np.stack([eval_1, eval_2], axis=1)
    pred_eigenvalues, pred_eigenvectors = np.linalg.eigh(y_pred_resized)

    # true_evals = self._calc_evals(y_true_resized)

    ####
    ## Eigenvalues of true/target tensors
    ####
    eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
              np.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                         4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                              y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

    eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
              np.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                         4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                              y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

    true_evals = np.stack([eval_1, eval_2], axis=1)

    # print("pred evals ", pred_evals)
    # print("true evals ", true_evals)
    # print("pred_evals - true_evals ", pred_evals - true_evals)

    true_eigenvalues, true_eigenvectors = np.linalg.eigh(y_true_resized)

    # Calculate MSE for eigenvalues
    #evals_mse = torch.mean((pred_evals - true_evals) ** 2)

    squared_err_k = []
    std_tar_k = []
    r2_k = []
    mse_k = []
    rmse_k = []
    nrmse_k = []

    for i in range(true_evals.shape[1]):
        targets = np.squeeze(true_evals[:, i, ...])
        predictions = np.squeeze(pred_evals[:, i, ...])
        squared_err_k.append((targets - predictions) ** 2)
        std_tar_k.append(np.std(targets))
        r2_k.append(1 - (np.sum(squared_err_k[i]) /
                         np.sum((targets - np.mean(targets)) ** 2)))
        mse_k.append(np.mean(squared_err_k[i]))
        rmse_k.append(np.sqrt(mse_k[i]))
        nrmse_k.append(rmse_k[i] / std_tar_k[i])


    all_pred_evec_1 = []
    all_pred_evec_2 = []
    all_true_evec_1 = []
    all_true_evec_2 = []
    for i in range(len(pred_eigenvalues)):
        linalg_pred_evals = pred_eigenvalues[i]
        linalg_true_evals = true_eigenvalues[i]
        linalg_pred_evecs = pred_eigenvectors[i]
        linalg_true_evecs = true_eigenvectors[i]

        pred_evec_1 = linalg_pred_evecs[:, np.argmin(np.abs(linalg_pred_evals - pred_evals[i][0]))]
        pred_evec_2 = linalg_pred_evecs[:, np.argmin(np.abs(linalg_pred_evals - pred_evals[i][1]))]

        all_pred_evec_1.append(pred_evec_1)
        all_pred_evec_2.append(pred_evec_2)

        true_evec_1 = linalg_true_evecs[:, np.argmin(np.abs(linalg_true_evals - true_evals[i][0]))]
        true_evec_2 = linalg_true_evecs[:, np.argmin(np.abs(linalg_true_evals - true_evals[i][1]))]

        all_true_evec_1.append(true_evec_1)
        all_true_evec_2.append(true_evec_2)


    all_true_evec_1 = np.array(all_true_evec_1)
    all_true_evec_2 = np.array(all_true_evec_2)
    all_pred_evec_1 = np.array(all_pred_evec_1)
    all_pred_evec_2 = np.array(all_pred_evec_2)


    # squared_err_k = []
    # std_tar_k = []
    # r2_k = []
    # mse_k = []
    # rmse_k = []
    # nrmse_k = []

    # print("all true evec 1 shape", all_true_evec_1.shape)
    # print("all true evec 2 shape", all_true_evec_2.shape)

    # targets = true_evec_1 #np.squeeze(targets_arr[:, i, ...])
    # predictions = pred_evec_1 #np.squeeze(predictions_arr[:, i, ...])
    # squared_err_evec_1 = (targets - predictions) ** 2
    # std_tar_evec_1 = np.std(targets)
    # r2_evec_1 = 1 - (np.sum(squared_err_evec_1) /
    #                  np.sum((targets - np.mean(targets)) ** 2))
    # mse_evec_1 = np.mean(squared_err_evec_1)
    # rmse_evec_1 = np.sqrt(mse_evec_1)
    # nrmse_evec_1 = rmse_evec_1 / std_tar_evec_1


    squared_err_evec_1 = []
    std_tar_evec_1 = []
    r2_evec_1 = []
    mse_evec_1 = []
    rmse_evec_1 = []
    nrmse_evec_1 = []

    for i in range(all_true_evec_1.shape[1]):
        targets_evec_1 = np.squeeze(all_true_evec_1[:, i, ...])
        predictions_evec_1 = np.squeeze(all_pred_evec_1[:, i, ...])
        squared_err_evec_1.append((targets_evec_1 - predictions_evec_1) ** 2)
        std_tar_evec_1.append(np.std(targets_evec_1))
        r2_evec_1.append(1 - (np.sum(squared_err_evec_1[i]) /
                         np.sum((targets_evec_1 - np.mean(targets_evec_1)) ** 2)))
        mse_evec_1.append(np.mean(squared_err_evec_1[i]))
        rmse_evec_1.append(np.sqrt(mse_evec_1[i]))
        nrmse_evec_1.append(rmse_evec_1[i] / std_tar_evec_1[i])

    squared_err_evec_2 = []
    std_tar_evec_2 = []
    r2_evec_2 = []
    mse_evec_2 = []
    rmse_evec_2 = []
    nrmse_evec_2 = []

    for i in range(all_true_evec_2.shape[1]):
        targets_evec_2 = np.squeeze(all_true_evec_2[:, i, ...])
        predictions_evec_2 = np.squeeze(all_pred_evec_2[:, i, ...])
        squared_err_evec_2.append((targets_evec_2 - predictions_evec_2) ** 2)
        std_tar_evec_2.append(np.std(targets_evec_2))
        r2_evec_2.append(1 - (np.sum(squared_err_evec_2[i]) /
                              np.sum((targets_evec_2 - np.mean(targets_evec_2)) ** 2)))
        mse_evec_2.append(np.mean(squared_err_evec_2[i]))
        rmse_evec_2.append(np.sqrt(mse_evec_2[i]))
        nrmse_evec_2.append(rmse_evec_2[i] / std_tar_evec_2[i])


    # targets = true_evec_2  # np.squeeze(targets_arr[:, i, ...])
    # predictions = pred_evec_2  # np.squeeze(predictions_arr[:, i, ...])
    # squared_err_evec_2 = (targets - predictions) ** 2
    # std_tar_evec_2 = np.std(targets)
    # r2_evec_2 = 1 - (np.sum(squared_err_evec_2) /
    #                  np.sum((targets - np.mean(targets)) ** 2))
    # mse_evec_2 = np.mean(squared_err_evec_2)
    # rmse_evec_2 = np.sqrt(mse_evec_2)
    # nrmse_evec_2 = rmse_evec_2 / std_tar_evec_2


    print("EVALS: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_k, r2_k, rmse_k, nrmse_k))
    print("EVEC_1: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_1, r2_evec_1, rmse_evec_1, nrmse_evec_1))
    print("EVEC_2: mse: {}, r2: {}, rmse: {}, nrmse: {}".format(mse_evec_2, r2_evec_2, rmse_evec_2, nrmse_evec_2))

    #return mse_k, rmse_k, nrmse_k, r2_k


class FrobeniusNorm(nn.Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()

    def forward(self, y_pred, y_true):
        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = y_pred[:, 1, ...] - y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        #k_xx = torch.mean(k_xx ** 2)
        #k_xy = torch.mean(k_xy ** 2)
        #k_yy = torch.mean(k_yy ** 2)

        return torch.mean(torch.sqrt(k_xx**2 + 2 * k_xy**2 + k_yy**2))

class FrobeniusNorm2(nn.Module):
    def __init__(self):
        super(FrobeniusNorm2, self).__init__()

    def forward(self, y_pred, y_true):
        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = y_pred[:, 1, ...] - y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        k_xx = torch.mean(k_xx ** 2)
        k_xy = torch.mean(k_xy ** 2)
        k_yy = torch.mean(k_yy ** 2)

        return torch.sqrt(k_xx**2 + 2 * k_xy**2 + k_yy**2)


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
    def forward(self, y_pred, y_true):
        criterion = nn.CosineSimilarity()
        return 1 - criterion(y_pred, y_true)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = torch.mean(self.weights * mse_loss)
        return weighted_mse_loss


class MSELossLargeEmph(nn.Module):
    def __init__(self, params):
        super(MSELossLargeEmph, self).__init__()
        #print("params ", params)
        self.weights_min_abs = torch.Tensor(params[0])
        self.mult_coef = params[1]
        self.fce = params[2]
        if torch.cuda.is_available():
            self.weights_min_abs = self.weights_min_abs.cuda()
            #self.mult_coef = self.mult_coef.cuda()

        #print("weights min abs ", self.weights_min_abs)
        #print("self.mult_coef ", self.mult_coef)

    def forward(self, y_pred, y_true):
        if str(y_true.device) == "cpu":
            self.weights_min_abs = self.weights_min_abs.cpu()

        error = y_true - y_pred
        #print("y_true ", len(y_true.shape))
        weights = (y_true + self.weights_min_abs) * self.mult_coef + 1

        if self.fce == "2":
            weights = weights ** 2
        elif self.fce == "3":
            weights = weights ** 3
        elif self.fce == "4":
            weights = weights ** 4
        elif self.fce == "exp":
            weights = torch.exp(weights)
        elif self.fce == "exp_2":
            weights = torch.exp(weights)**2

        if len(y_true.shape) == 1:
            weights[1] = 1
        else:
            weights[:, 1] = 1

        weighted_error = torch.square(error) * weights
        #print("weighted error ", weighted_error)
        #print("y_true: {}, squared error: {}, weighted error: {}".format(y_true, torch.square(error), weighted_error))
        return torch.mean(weighted_error)


class MSELossLargeEmphAvg(nn.Module):
    def __init__(self, params):
        super(MSELossLargeEmphAvg, self).__init__()
        #print("params ", params)
        self.weights_min_abs = torch.Tensor(params[0])
        self.mult_coef =  params[1]
        self.fce = params[2]
        if torch.cuda.is_available():
            self.weights_min_abs = self.weights_min_abs.cuda()
            #self.mult_coef = self.mult_coef.cuda()

        # print("weights min abs ", self.weights_min_abs)
        # print("self.mult_coef ", self.mult_coef)

    def forward(self, y_pred, y_true):
        if str(y_true.device) == "cpu":
            self.weights_min_abs = self.weights_min_abs.cpu()

        error = y_true - y_pred
        #print("y_true ", y_true)
        weights = (y_true + self.weights_min_abs) * self.mult_coef + 1

        if self.fce == "2":
            weights = weights ** 2
        elif self.fce == "3":
            weights = weights ** 3
        elif self.fce == "4":
            weights = weights ** 4
        elif self.fce == "exp":
            weights = torch.exp(weights)
        elif self.fce == "exp_2":
            weights = torch.exp(weights)**2

        if len(y_true.shape) == 1:
            weights[1] = (weights[0] + weights[1])/2
        else:
            weights[:, 1] = weights[:, [0, 2]].mean(dim=1)

        #print("wieghts ", weights)
        weighted_error = torch.square(error) * weights
        #print("weighted error ", weighted_error)
        #print("y_true: {}, squared error: {}, weighted error: {}".format(y_true, torch.square(error), weighted_error))
        return torch.mean(weighted_error)


class WeightedMSELossSum(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELossSum, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()

        #print("mse loss ", len(mse_loss.shape))

        if len(mse_loss.shape) == 1:
            mse_loss_sum = self.weights * mse_loss
        else:
            mse_loss_sum = torch.sum(self.weights * mse_loss, dim=1)
        #print("mse loss sum ", mse_loss_sum)
        weighted_mse_loss = torch.mean(mse_loss_sum)
        return weighted_mse_loss


class RelMSELoss(nn.Module):
    def __init__(self, weights):
        super(RelMSELoss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        # k_xx_mse = F.mse_loss(y_pred[:, 0, ...], y_true[:, 0, ...])
        # print("k xx mse ", k_xx_mse)
        # k_xx_rmse = torch.sqrt(k_xx_mse / (y_true[:, 0, ...] ** 2))
        #
        # k_xy_mse = F.mse_loss(y_pred[:, 1, ...], y_true[:, 1, ...])
        # k_xy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 1, ...] ** 2))
        #
        # k_yy_mse = F.mse_loss(y_pred[:, 2, ...], y_true[:, 2, ...])
        # k_yy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 2, ...] ** 2))
        #print("y_pred[:, 0, ...] ", y_pred[:, 0, ...])
        #print("y_true[:, 0, ...] ", y_true[:, 0, ...])
        #
        #print("(y_pred[:, 0, ...] - y_true[:, 0, ...]) ", (y_pred[:, 0, ...] - y_true[:, 0, ...]))


        k_xx = torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...]))/torch.square(y_true[:, 0, ...])
        k_xy = torch.square(y_pred[:, 1, ...] - y_true[:, 1, ...])/torch.square(y_true[:, 1, ...])
        k_yy = torch.square(y_pred[:, 2, ...] - y_true[:, 2, ...])/torch.square(y_true[:, 2, ...])

        #print("torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...])) ", torch.square((y_pred[:, 0, ...] - y_true[:, 0, ...])))
        #print("torch.square(y_true[:, 0, ...]) ", torch.square(y_true[:, 0, ...]))

        #print("k xx ", k_xx)
        # print("k xy ", k_xy)
        # print("k yy ", k_yy)

        rel_mse = torch.mean(k_xx) + torch.mean(k_xy) + torch.mean(k_yy)#torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)
        #total_loss = k_xx_rmse + k_xy_rmse + k_yy_rmse
        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx), torch.mean(k_xy), torch.mean(k_yy)))
        #print("total loss ", total_loss)
        #exit()

        return rel_mse


class MASELoss(nn.Module):
    def __init__(self, weights):
        super(MASELoss, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        numerator = torch.abs(y_true[:, 0, ...] - y_pred[:, 0, ...]).mean()
        denominator = torch.abs(y_true[1:, 0, ...] - y_true[:-1, 0, ...]).mean()
        #print("numerator kxx", numerator)
        #print("denomination kxx", denominator)

        # Calculate MASE
        mase_k_xx = numerator / denominator

        numerator = torch.abs(y_true[:, 1, ...] - y_pred[:, 1, ...]).mean()
        denominator = torch.abs(y_true[1:, 1, ...] - y_true[:-1, 1, ...]).mean()
        #print("numerator kxy", numerator)
        #print("denomination kxy ", denominator)

        # Calculate MASE
        mase_k_xy = numerator / denominator

        numerator = torch.abs(y_true[:, 2, ...] - y_pred[:, 2, ...]).mean()
        denominator = torch.abs(y_true[1:, 2, ...] - y_true[:-1, 2, ...]).mean()
        #print("numerator kyy", numerator)
        #print("denomination kyy ", denominator)

        # Calculate MASE
        mase_k_yy = numerator / denominator

        #print("mase k xx ", mase_k_xx)
        #print("mase k xy ", mase_k_xy)
        #print("mase k yy ", mase_k_yy)

        total_mase = mase_k_xx + mase_k_xy + mase_k_yy

        print("MASE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(total_mase, mase_k_xx, mase_k_xy, mase_k_yy))

        return total_mase


class RelMSELoss2(nn.Module):
    def __init__(self, weights):
        super(RelMSELoss2, self).__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred.unsqueeze(0)
        #     y_true = y_true.unsqueeze(0)

        # k_xx_mse = F.mse_loss(y_pred[:, 0, ...], y_true[:, 0, ...])
        # print("k xx mse ", k_xx_mse)
        # k_xx_rmse = torch.sqrt(k_xx_mse / (y_true[:, 0, ...] ** 2))
        #
        # k_xy_mse = F.mse_loss(y_pred[:, 1, ...], y_true[:, 1, ...])
        # k_xy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 1, ...] ** 2))
        #
        # k_yy_mse = F.mse_loss(y_pred[:, 2, ...], y_true[:, 2, ...])
        # k_yy_rmse = torch.sqrt(k_xx_mse / (y_true[:, 2, ...] ** 2))
        print("y_pred[:, 0, ...] ", y_pred[:, 0, ...])
        print("y_true[:, 0, ...] ", y_true[:, 0, ...])
        print("(y_pred[:, 0, ...]/y_true[:, 0, ...]) ", (y_pred[:, 0, ...] /y_true[:, 0, ...]))
        print("(y_pred[:, 0, ...]/y_true[:, 0, ...]) -1  ", (y_pred[:, 0, ...] / y_true[:, 0, ...])-1)

        k_xx = torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) - 1)
        k_xy = torch.square((y_pred[:, 1, ...]/y_true[:, 1, ...]) - 1)
        k_yy = torch.square((y_pred[:, 2, ...]/y_true[:, 2, ...]) - 1)

        print("torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) -1 ) ", torch.square((y_pred[:, 0, ...]/y_true[:, 0, ...]) - 1))
        #print("torch.square(y_true[:, 0, ...]) ", torch.square(y_true[:, 0, ...]))

        print("k xx ", k_xx)
        # print("k xy ", k_xy)
        # print("k yy ", k_yy)

        rel_mse = torch.mean(k_xx) + torch.mean(k_xy) + torch.mean(k_yy)#torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)
        #total_loss = k_xx_rmse + k_xy_rmse + k_yy_rmse
        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx), torch.mean(k_xy), torch.mean(k_yy)))
        #print("total loss ", total_loss)
        #exit()

        return rel_mse


class RelXYMSELoss(nn.Module):
    def __init__(self, weights):
        super(RelXYMSELoss, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):

        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        k_xx = y_pred[:, 0, ...] - y_true[:, 0, ...]
        k_xy = ((y_pred[:, 1, ...] - y_true[:, 1, ...]))/y_true[:, 1, ...]
        k_yy = y_pred[:, 2, ...] - y_true[:, 2, ...]

        rel_mse = torch.mean(k_xx ** 2 + k_xy ** 2 + k_yy ** 2)

        print("RelMSE: {},  k_xx**2: {},  k_xy**2: {},  k_yy**2: {}".format(rel_mse, torch.mean(k_xx ** 2),
                                                                            torch.mean(k_xy ** 2),
                                                                            torch.mean(k_yy ** 2)))

        return rel_mse


class EighMSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def _calc_evals(self, batch_data):
        eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        evals = torch.stack([eval_1, eval_2], axis=1)

        return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_evec_1 = 0
        mse_evec_2 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                # Calculate MSE for eigenvectors
                mse_evec_1 += torch.mean((pred_evec_1 - true_evec_1) ** 2)
                mse_evec_2 += torch.mean((pred_evec_2 - true_evec_2) ** 2)

            total_evec_mse = self.weights[1] * mse_evec_1 + self.weights[2] * mse_evec_2
            mse = self.weights[0] * evals_mse + (total_evec_mse / (i+1))

            print("MSE evals: {}, evec1: {}, evec2: {}, total evec mse / batch size : {}".format(evals_mse, mse_evec_1,
                                                                                                 mse_evec_2,
                                                                                                 total_evec_mse / (i+1)))

        return mse



class EighMSE_2(nn.Module):
    def __init__(self, weights):
        super(EighMSE_2, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    # def _calc_evals(self, batch_data):
    #     eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     evals = torch.stack([eval_1, eval_2], axis=1)
    #
    #     return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_evec_1_0 = 0
        mse_evec_1_1 = 0
        mse_evec_2_0 = 0
        mse_evec_2_1 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            all_pred_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_pred_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                all_pred_evec_1[i] = pred_evec_1
                all_pred_evec_2[i] = pred_evec_2

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                all_true_evec_1[i] = true_evec_1
                all_true_evec_2[i] = true_evec_2

            mse_evec_1_0 = torch.mean((all_pred_evec_1[:, 0] - all_true_evec_1[:, 0]) ** 2)
            mse_evec_1_1 = torch.mean((all_pred_evec_1[:, 1] - all_true_evec_1[:, 1]) ** 2)

            mse_evec_2_0 = torch.mean((all_pred_evec_2[:, 0] - all_true_evec_2[:, 0]) ** 2)
            mse_evec_2_1 = torch.mean((all_pred_evec_2[:, 1] - all_true_evec_2[:, 1]) ** 2)


            total_evec_mse = self.weights[1] * mse_evec_1_0 + self.weights[2] * mse_evec_1_1 + self.weights[3] * mse_evec_2_0 + self.weights[4] * mse_evec_2_1
            mse = self.weights[0] * evals_mse + total_evec_mse

            print("MSE evals: {}, evec_1_0: {}, evec_1_1: {},  evec_1_0: {}, evec_1_1: {}".format(evals_mse, mse_evec_1_0,mse_evec_1_1,
                                                                                                 mse_evec_2_0, mse_evec_2_1,
                                                                                                 ))

        return mse


class EighMSE_2_MSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE_2_MSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    # def _calc_evals(self, batch_data):
    #     eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
    #                    torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
    #                               4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
    #                                    batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2
    #
    #     evals = torch.stack([eval_1, eval_2], axis=1)
    #
    #     return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_loss = F.mse_loss(y_pred, y_true)
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = self.weights[5] * mse_loss

        mse_evec_1_0 = 0
        mse_evec_1_1 = 0
        mse_evec_2_0 = 0
        mse_evec_2_1 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}".format(mse))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            all_pred_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_pred_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_1 = torch.empty(len(pred_eigenvalues), 2)
            all_true_evec_2 = torch.empty(len(pred_eigenvalues), 2)
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                all_pred_evec_1[i] = pred_evec_1
                all_pred_evec_2[i] = pred_evec_2

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                all_true_evec_1[i] = true_evec_1
                all_true_evec_2[i] = true_evec_2

            mse_evec_1_0 = torch.mean((all_pred_evec_1[:, 0] - all_true_evec_1[:, 0]) ** 2)
            mse_evec_1_1 = torch.mean((all_pred_evec_1[:, 1] - all_true_evec_1[:, 1]) ** 2)

            mse_evec_2_0 = torch.mean((all_pred_evec_2[:, 0] - all_true_evec_2[:, 0]) ** 2)
            mse_evec_2_1 = torch.mean((all_pred_evec_2[:, 1] - all_true_evec_2[:, 1]) ** 2)


            total_evec_mse = self.weights[1] * mse_evec_1_0 + self.weights[2] * mse_evec_1_1 + self.weights[3] * mse_evec_2_0 + self.weights[4] * mse_evec_2_1
            mse = self.weights[0] * evals_mse + total_evec_mse

            print("MSE evals: {}, evec_1_0: {}, evec_1_1: {},  evec_1_0: {}, evec_1_1: {}, data MSE: {}".format(evals_mse, mse_evec_1_0,mse_evec_1_1,
                                                                                                 mse_evec_2_0, mse_evec_2_1, weighted_mse_loss
                                                                                                 ))
        final_mse = mse + weighted_mse_loss
        return final_mse


class EighMSE_MSE(nn.Module):
    def __init__(self, weights):
        super(EighMSE_MSE, self).__init__()
        print("weights ", weights)
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def _calc_evals(self, batch_data):
        eval_1 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) +
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        eval_2 = ((batch_data[:, 0, 0] + batch_data[:, 1, 1]) -
                       torch.sqrt((-batch_data[:, 0, 0] - batch_data[:, 1, 1]) ** 2 -
                                  4 * (batch_data[:, 0, 0] * batch_data[:, 1, 1] -
                                       batch_data[:, 0, 1] * batch_data[:, 1, 0]))) / 2

        evals = torch.stack([eval_1, eval_2], axis=1)

        return evals

    def forward(self, y_pred, y_true):
        if len(y_true.shape) == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

        #tensor_loss = F.mse_loss(y_pred, y_true)

        y_pred_resized = torch.empty(y_pred.shape[0], 2, 2)
        y_pred_resized[:, 0, 0] = y_pred[:, 0]
        y_pred_resized[:, 0, 1] = y_pred[:, 1]
        y_pred_resized[:, 1, 0] = y_pred[:, 1]
        y_pred_resized[:, 1, 1] = y_pred[:, 2]

        y_true_resized = torch.empty(y_true.shape[0], 2, 2)
        y_true_resized[:, 0, 0] = y_true[:, 0]
        y_true_resized[:, 0, 1] = y_true[:, 1]
        y_true_resized[:, 1, 0] = y_true[:, 1]
        y_true_resized[:, 1, 1] = y_true[:, 2]

        #pred_evals = self._calc_evals(y_pred_resized)

        ####
        ## Eigenvalues of predicted tensors
        ####
        eval_1 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) +
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        eval_2 = ((y_pred_resized[:, 0, 0] + y_pred_resized[:, 1, 1]) -
                  torch.sqrt((-y_pred_resized[:, 0, 0] - y_pred_resized[:, 1, 1]) ** 2 -
                             4 * (y_pred_resized[:, 0, 0] * y_pred_resized[:, 1, 1] -
                                  y_pred_resized[:, 0, 1] * y_pred_resized[:, 1, 0]))) / 2

        pred_evals = torch.stack([eval_1, eval_2], axis=1)
        pred_eigenvalues, pred_eigenvectors = torch.linalg.eigh(y_pred_resized)

        #true_evals = self._calc_evals(y_true_resized)

        ####
        ## Eigenvalues of true/target tensors
        ####
        eval_1 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) +
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        eval_2 = ((y_true_resized[:, 0, 0] + y_true_resized[:, 1, 1]) -
                  torch.sqrt((-y_true_resized[:, 0, 0] - y_true_resized[:, 1, 1]) ** 2 -
                             4 * (y_true_resized[:, 0, 0] * y_true_resized[:, 1, 1] -
                                  y_true_resized[:, 0, 1] * y_true_resized[:, 1, 0]))) / 2

        true_evals = torch.stack([eval_1, eval_2], axis=1)

        # print("pred evals ", pred_evals)
        # print("true evals ", true_evals)
        # print("pred_evals - true_evals ", pred_evals - true_evals)

        true_eigenvalues, true_eigenvectors = torch.linalg.eigh(y_true_resized)

        # Calculate MSE for eigenvalues
        evals_mse = torch.mean((pred_evals - true_evals) ** 2)

        mse_loss = F.mse_loss(y_pred, y_true)
        if str(mse_loss.device) == "cpu":
            self.weights = self.weights.cpu()
        weighted_mse_loss = self.weights[3] * mse_loss

        mse_evec_1 = 0
        mse_evec_2 = 0
        if self.weights[1] == 0 and self.weights[1] == 0:
            mse = self.weights[0] * evals_mse
            print("MSE evals: {}, data mse: {}".format(mse, weighted_mse_loss))
            #print("MSE evals: {}, tensor MSE: {}".format(mse, tensor_loss))
        else:
            for i in range(len(pred_eigenvalues)):
                linalg_pred_evals = pred_eigenvalues[i]
                linalg_true_evals = true_eigenvalues[i]
                linalg_pred_evecs = pred_eigenvectors[i]
                linalg_true_evecs = true_eigenvectors[i]

                pred_evec_1 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][0]))]
                pred_evec_2 = linalg_pred_evecs[:, torch.argmin(torch.abs(linalg_pred_evals - pred_evals[i][1]))]

                true_evec_1 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][0]))]
                true_evec_2 = linalg_true_evecs[:, torch.argmin(torch.abs(linalg_true_evals - true_evals[i][1]))]

                # Calculate MSE for eigenvectors
                mse_evec_1 += torch.mean((pred_evec_1 - true_evec_1) ** 2)
                mse_evec_2 += torch.mean((pred_evec_2 - true_evec_2) ** 2)

            total_evec_mse = self.weights[1] * mse_evec_1 + self.weights[2] * mse_evec_2
            mse = self.weights[0] * evals_mse + (total_evec_mse / (i+1))

            print("MSE evals: {}, evec1: {}, evec2: {}, data mse: {} total evec mse / batch size : {}".format(evals_mse, mse_evec_1,
                                                                                                 mse_evec_2, weighted_mse_loss,
                                                                                                 total_evec_mse / (i+1)))


        final_mse = mse + weighted_mse_loss
        return final_mse



class LossX(nn.Module):
    """
    Parameters
    ----------
    num_attrs_X : int
        Number of node attributes.
    """
    def __init__(self,
                 num_attrs_X):
        super().__init__()

        self.num_attrs_X = num_attrs_X

    def forward(self, true_X, logit_X):
        """
        Parameters
        ----------
        true_X : torch.Tensor of shape (F, |V|, 2)
            X_one_hot_3d[f, :, :] is the one-hot encoding of the f-th node attribute.
        logit_X : torch.Tensor of shape (|V|, F, 2)
            Predicted logits for the node attributes.

        Returns
        -------
        loss_X : torch.Tensor
            Scalar representing the loss for node attributes.
        """
        true_X = true_X.transpose(0, 1)               # (|V|, F, 2)
        # v1x1, v1x2, ..., v1xd, v2x1, ...
        true_X = true_X.reshape(-1, true_X.size(-1))  # (|V| * F, 2)

        # v1x1, v1x2, ..., v1xd, v2x1, ...
        logit_X = logit_X.reshape(true_X.size(0), -1) # (|V| * F, 2)

        true_X = torch.argmax(true_X, dim=-1)         # (|V| * F)
        loss_X = F.cross_entropy(logit_X, true_X)

        return loss_X


class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_G, logit_G):
        loss = F.cross_entropy(logit_G, true_G)
        return loss


def get_loss_fn(loss_function):
    loss_fn_name = loss_function[0]
    loss_fn_params = loss_function[1]
    if loss_fn_name == "MSE" or loss_fn_name == "L2":
        return nn.MSELoss()
    elif loss_fn_name == "L1":
        return nn.L1Loss()
    elif loss_fn_name == "Frobenius":
        return FrobeniusNorm()
    elif loss_fn_name == "Frobenius2":
        return FrobeniusNorm2()
    elif loss_fn_name == "MSEweighted":
        return WeightedMSELoss(loss_fn_params)
    elif loss_fn_name == "RelMSELoss":
        return RelMSELoss(loss_fn_params)
    elif loss_fn_name == "MASELoss":
        return MASELoss(loss_fn_params)
    elif loss_fn_name == "RelMSELoss2":
        return RelMSELoss2(loss_fn_params)
    elif loss_fn_name == "RelXYMSELoss":
        return RelXYMSELoss(loss_fn_params)
    elif loss_fn_name == "MSEweightedSum":
        return WeightedMSELossSum(loss_fn_params)
    elif loss_fn_name == "EighMSE":
        return EighMSE(loss_fn_params)
    elif loss_fn_name == "EighMSE_2":
        return EighMSE_2(loss_fn_params)
    elif loss_fn_name == "EighMSE_MSE":
        return EighMSE_MSE(loss_fn_params)
    elif loss_fn_name == "EighMSE_2_MSE":
        return EighMSE_2_MSE(loss_fn_params)
    elif loss_fn_name == "MSELossLargeEmph":
        return MSELossLargeEmph(loss_fn_params)
    elif loss_fn_name == "MSELossLargeEmphAvg":
        return MSELossLargeEmphAvg(loss_fn_params)
    elif loss_fn_name == "KL" or loss_fn_name == "cross_entropy":
        return KLLoss()


