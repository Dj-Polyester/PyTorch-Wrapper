class Debug:
    printEnabled = True

    class TypeError(TypeError):
        def __init__(self, **kwargs) -> None:
            kwargsValues = kwargs.values()
            types = ", ".join(map(lambda x: x.__class__.__name__, kwargsValues))
            names = ", ".join(map(str, kwargs.keys()))
            super().__init__(f"Variable(s) {names} have invalid type(s) {types}")

    class ValueError(ValueError):
        def __init__(self, **kwargs) -> None:
            values = ", ".join(map(str, kwargs.values()))
            names = ", ".join(map(str, kwargs.keys()))
            super().__init__(f"Variable(s) {names} have invalid value(s) {values}")

    class AttributeError(AttributeError):
        def __init__(self, **kwargs) -> None:
            attrs = ", ".join(map(str, kwargs.values()))
            objs = ", ".join(map(str, kwargs.keys()))
            super().__init__(f"Object(s) {objs} have invalid attribute(s) {attrs}")

    def Print(**kwargs):
        if Debug.printEnabled:
            print(", ".join([f"{k}: {v}" for k, v in kwargs.items()]))
