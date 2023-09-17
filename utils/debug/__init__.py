printEnabled = True


def TypeError(**kwargs):
    kwargsValues = kwargs.values()
    types = ", ".join(map(lambda x: x.__class__.__name__, kwargsValues))
    names = ", ".join(map(str, kwargs.keys()))
    return TypeError(f"Variable(s) {names} have invalid type(s) {types}")


def ValueError(**kwargs):
    values = ", ".join(map(str, kwargs.values()))
    names = ", ".join(map(str, kwargs.keys()))
    return ValueError(f"Variable(s) {names} have invalid value(s) {values}")


def Print(**kwargs):
    if printEnabled:
        print(", ".join([f"{k}: {v}" for k, v in kwargs.items()]))
