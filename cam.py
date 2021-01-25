import numpy as np


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = (output.cpu()).data.numpy()

    def remove(self):
        self.hook.remove()


def calculate_cam(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def extract_cam(model, feature_layer_name, fc_layer_name, sample):
    feature_layer = model._modules.get(feature_layer_name)
    activated_features = SaveFeatures(feature_layer)

    model.eval()
    (x1, x2, y) = sample
    out = model(x1, x2)
    pred_probabilities = out.data.squeeze()
    probs, idx = pred_probabilities.sort(0, True)
    activated_features.remove()

    weight_softmax_params = list(model._modules.get(fc_layer_name).parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    for i in range(0, 3):
        line = '{:.3f} -> {}'.format(probs[i], idx[i].item())
        print(line)

    cam = calculate_cam(activated_features.features, weight_softmax, [idx[0].item()])



