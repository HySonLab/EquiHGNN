class Registry:
    mapping = {"model_name_mapping": {}, "data_name_mapping": {}}

    @classmethod
    def register_model(cls, name: str):
        """Register a model with key"""

        def wrapper(model_cls):
            if name in cls.mapping["model_name_mapping"]:
                raise ValueError(f"Class with name {name} already registered.")
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrapper

    @classmethod
    def register_data(cls, name: str):
        """Register a data with key"""

        def wrapper(data_cls):
            if name in cls.mapping["data_name_mapping"]:
                raise ValueError(f"Class with name {name} already registered.")
            cls.mapping["data_name_mapping"][name] = data_cls
            return data_cls

        return wrapper

    @classmethod
    def get_model_class(cls, name: str):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_data_class(cls, name: str):
        return cls.mapping["data_name_mapping"].get(name, None)

    @classmethod
    def list_out(cls):
        return cls.mapping


registry = Registry()
