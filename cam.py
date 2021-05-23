import numpy as np
import torch
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = (output.cpu()).data.numpy()

    def remove(self):
        self.hook.remove()


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

    class_id = idx[0].item()
    feature_conv = activated_features.features
    batch_size, channels_count, channel_size = feature_conv.shape
    cam = weight_softmax[class_id].dot(feature_conv.reshape((channels_count, -1)))
    cam = cam - np.min(cam)
    return cam


def extract_grad_cam(model, sample):
    (x1, x2, y) = sample
    if torch.cuda.is_available():
        x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

    model.eval()
    out = model(x1, x2)

    predicted_class = out.cpu().detach().squeeze().numpy().argmax()
    # we are going to do the back-propagation with the logit of specific class
    out[:, predicted_class].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2])

    # get the activations of the last convolutional layer
    activations = model.get_activations(x1, x2).detach()

    # weight the channels by corresponding gradients
    for i in range(96):
        activations[:, i, :] *= pooled_gradients[i]

    # average the channels of the activations
    grad_cam = torch.mean(activations, dim=1).squeeze().numpy()
    grad_cam = grad_cam - np.min(grad_cam)
    return grad_cam


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
    fig, axs = plt.subplots(channels_number, figsize=(120, 80), dpi=50)
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