from equihgnn.common.registry import registry
from equihgnn.models import *  # noqa: F403, F401


def create_model(model_name: str):
    model_cls = registry.get_model_class(name=model_name)

    if model_cls is None:
        raise ValueError(f"Model with name {model_name} not found.")
    return model_cls


def create_data(data_name: str):
    data_cls = registry.get_data_class(name=data_name)

    if data_cls is None:
        raise ValueError(f"Data with name {data_name} not found.")
    return data_cls
