import torch

from cam import extract_cam, extract_grad_cam
from data_manager import DataManager
from explainable_nn import ExplainableNN, SimpleExplainableNN, GradSimpleExplainableNN, RegularizedGradSimpleExplainableNN


def get_model(model_name=None):
    # model = ExplainableNN()   # CAM-based architecture, too difficult for training on default PC
    # model = SimpleExplainableNN() # CAM-based architecture
    # model = GradSimpleExplainableNN() # Grad-CAM-based architecture
    model = RegularizedGradSimpleExplainableNN()  # Grad-CAM-based architecture + mode dropouts

    # loading already saved model
    if model_name is not None:
        model.load_state_dict(torch.load(f'models/${model_name}'))

    return model


def get_data_manager():
    data_manager = DataManager(
        train_dir='data/train',
        test_dir='data/test',
        labels_file='labels.csv',
        data_folder='samples',
        batch_size=256
    )
    return data_manager


def get_cam_extractor():
    # only for CAM-based architectures
    # cam_extractor = extract_cam

    # only for Grad-CAM-based architectures
    cam_extractor = extract_grad_cam

    return cam_extractor


training_epochs_count = 10
