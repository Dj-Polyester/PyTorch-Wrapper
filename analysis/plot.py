import debug
from ..tuning import Tuner
from ..tuning.metrics import Metrics
import matplotlib.pyplot as plt
from types import FunctionType


def _convert(item: str):
    try:
        return int(item)
    except ValueError:
        try:
            return float(item)
        except ValueError:
            try:
                return Str(item).tolist()
            except ValueError:
                try:
                    return Str(item).toset()
                except ValueError:
                    try:
                        return Str(item).todict()
                    except ValueError:
                        return item


class Str(str):
    def initElementalContainer(self):
        self.tmpContainer = self.type()

    def addFn(self, item):
        if isinstance(self.tmpContainer, list):
            self.tmpContainer.insert(0, item)
        if isinstance(self.tmpContainer, set):
            if isinstance(item, set):
                item = frozenset(item)
            self.tmpContainer.add(item)
        if isinstance(self.tmpContainer, dict):
            k, v = item
            self.tmpContainer[k] = v

    def tolist(self, fn=_convert):
        return self.tocontainer(list, fn)

    def toset(self, fn=_convert):
        return self.tocontainer(set, fn)

    def todict(self, fn=_convert):
        return self.tocontainer(dict, fn)

    def _append2stack(self, fn):
        if self.type == dict:
            self.itemPair.append(self.itemTmp)
            self.itemPair = tuple(map(fn, self.itemPair))
            self.stack.append(self.itemPair)
            self.itemPair = []
        else:
            self.stack.append(fn(self.itemTmp))
        self.itemTmp = ""

    def tocontainer(
        self,
        _type,
        fn=_convert,
    ):
        if _type == list:
            openingSym = "["
            closingSym = "]"
        elif _type == set or _type == dict:
            openingSym = "{"
            closingSym = "}"

        self.type = _type
        self.stack = []
        self.itemTmp = ""
        prevSymComma = False
        self.itemPair = []
        for c in self:
            if c == openingSym:
                self.itemTmp = ""
                self.stack.append(openingSym)
            elif c == ",":
                if prevSymComma:
                    raise Exception(f"Comma cannot come after comma")
                if self.itemTmp == "":
                    if self.stack[-1] == openingSym:
                        raise Exception(f"Comma cannot come after {openingSym}")
                else:
                    self._append2stack(fn)
            elif c == " ":
                pass
            elif c == ":":
                self.itemPair.append(self.itemTmp)
                self.itemTmp = ""
            elif c == closingSym:
                if self.itemTmp == "":
                    if self.stack[-1] != openingSym and (
                        prevSymComma or not isinstance(self.stack[-1], self.type)
                    ):
                        raise Exception(f"Comma cannot come before {closingSym}")
                else:
                    self._append2stack(fn)
                self.initElementalContainer()
                while self.stack[-1] != openingSym:
                    stackLast = self.stack.pop()
                    self.addFn(stackLast)
                    if len(self.stack) == 0:
                        raise Exception(
                            f"There are more {closingSym} than {openingSym}"
                        )
                self.stack.pop()
                self.stack.append(self.tmpContainer)
            else:
                self.itemTmp += c
            prevSymComma = c == "," or c == " "
        for elem in self.stack:
            if elem == openingSym:
                raise Exception(f"There are more {openingSym} than {closingSym}")
        if self.stack == []:
            raise ValueError(f"The given string {self} is not a list literal")
        return self.stack[0]

class Plotter:
    def plot2d(
        self,
        metrics: str | FunctionType | list | Metrics,
        tuner: Tuner = None,
        filename: str = None,
    ):
        # obtain metrics
        if tuner == None and filename == None:
            debug.ValueError(tuner=tuner, filename=filename)

        if (
            isinstance(metrics, str)
            or isinstance(metrics, FunctionType)
            or isinstance(metrics, list)
        ):
            self.metrics = Metrics(metrics)
        elif tuner == None:
            debug.ValueError(tuner=tuner, metrics=metrics)
        else:
            self.metrics = tuner.cv.model.metrics
        if filename != None:
            allMetrics = None
            scores = None
            with open(filename, "r") as f:
                allMetrics, scores = f.readlines()
            allMetrics = list(allMetrics)
            scores = 

        elif tuner != None:
            pass
