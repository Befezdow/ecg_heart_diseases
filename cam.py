import numpy as np
import torch
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = (output.cpu()).data.numpy()

    def remove(self):
        self.hook.remove()


def calculate_cam(feature_conv, weight_softmax, class_idx):
    batch_size, channels_count, channel_size = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((channels_count, -1)))
        cam = cam - np.min(cam)
        output_cam.append(cam)
    return output_cam


def extract_cam(model, feature_layer_name, fc_layer_name, sample):
    feature_layer = model._modules.get(feature_layer_name)
    activated_features = SaveFeatures(feature_layer)

    (x1, x2, y) = sample
    if torch.cuda.is_available():
        x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

    model.eval()
    out = model(x1, x2)
    pred_probabilities = out.data.squeeze()
    probs, idx = pred_probabilities.sort(0, True)
    activated_features.remove()

    weight_softmax_params = list(model._modules.get(fc_layer_name).parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    for i in range(0, 3):
        line = '{:.3f} -> {}'.format(probs[i], idx[i].item())
        print(line)

    return calculate_cam(activated_features.features, weight_softmax, [idx[0].item()])


def draw_cam(data_sample, data_cam):
    sample_timeseries_length = data_sample[1].shape[2]

    x_values = list(range(0, sample_timeseries_length))

    heat_values = []
    for x_value in x_values:
        cam_index = int(round(x_value / (sample_timeseries_length - 1) * (data_cam.shape[0] - 1)))
        heat_values.append(data_cam[cam_index])
    heat_func = interpolate.interp1d(x_values, heat_values, copy=False)
    heat_values = heat_func(x_values)

    min_cam_value = data_cam.min()
    max_cam_value = data_cam.max()

    channels_number = data_sample[1].shape[1]
    fig, axs = plt.subplots(channels_number)
    for i in range(0, channels_number):
        y_values = data_sample[1][0, i].tolist()

        y_min = data_sample[1][0, i].min()
        y_max = data_sample[1][0, i].max()

        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(min_cam_value, max_cam_value)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(heat_values)
        lc.set_linewidth(2)
        line = axs[i].add_collection(lc)
        fig.colorbar(line, ax=axs[i])

        axs[i].set_xlim(0, sample_timeseries_length)
        axs[i].set_ylim(y_min, y_max)

    plt.show()


def extract_grad_cam(model, sample):
    (x1, x2, y) = sample
    if torch.cuda.is_available():
        x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

    model.eval()
    out = model(x1, x2)

    # Now, we are going to do the back-propagation with the logit of
    # the 386th class which represents the ‘African_elephant’ in the ImageNet dataset.
    out[:, y.item()].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    gap = torch.nn.AvgPool1d(kernel_size=184)  # в качестве kernel_size берется размерность канала
    pooled_gradients = gap(gradients)

    # get the activations of the last convolutional layer
    activations = model.get_activations(x1, x2).detach()

    # TODO check formula
    # weight the channels by corresponding gradients
    # for i in range(512):
    #     activations[:, i, :, :] *= pooled_gradients[i]