from _io import TextIOWrapper


class Str(str):
    def __new__(cls, obj, limiter=None):
        instance = super().__new__(cls, obj)
        instance.whitespace = (" ", "\t", "\n")
        if isinstance(obj, TextIOWrapper):
            instance.file = obj
            instance.iter = lambda: instance._iter(limiter)
        else:
            instance.iter = instance.__iter__
        return instance

    @staticmethod
    def _convert(item: str):
        try:
            return int(item)
        except ValueError:
            try:
                return float(item)
            except ValueError:
                try:
                    return Str(item).tolist()
                except StrException:
                    try:
                        return Str(item).toset()
                    except StrException:
                        try:
                            return Str(item).todict()
                        except StrException:
                            try:
                                return eval(item)
                            except SyntaxError:
                                raise StrException(
                                    "Cannot convert atomic item to a valid object"
                                )

    def _addFn(self, item):
        if isinstance(self.tmpContainer, list):
            self.tmpContainer.insert(0, item)
        if isinstance(self.tmpContainer, set):
            if isinstance(item, set):
                item = frozenset(item)
            self.tmpContainer.add(item)
        if isinstance(self.tmpContainer, dict):
            if self.itemTmp == "":
                self.itemTmp = item
            else:
                self.tmpContainer[item] = self.itemTmp
                self.itemTmp = ""

    def _append2stack(self):
        self.stack.append(self.fn(self.itemTmp))
        self.itemTmp = ""

    def tolist(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(list, fn, allowCommaBeforeClosingSym)

    def toset(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(set, fn, allowCommaBeforeClosingSym)

    def todict(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(dict, fn, allowCommaBeforeClosingSym)

    def _processChar(self, c):
        if self.startString == None and c != " ":
            self.startString = c
            if self.startString != self.openingSym:
                raise StartsEndsException("starts", self.startString)
        if c == self.openingSym:
            if self.prevSym not in (" ", ",", self.openingSym) and (
                self.type != dict or self.prevSym != ":"
            ):
                raise InvalidCharException(self.prevSym, c)
            self.itemTmp = ""
            self.stack.append(self.openingSym)
        elif c == ",":
            if self.prevSym in (",", ":", self.openingSym):
                raise InvalidCharException(self.prevSym, c)
            if self.itemTmp != "":
                self._append2stack()
        elif c == ":":
            if self.prevSym in (",", ":", self.openingSym, self.closingSym):
                raise InvalidCharException(self.prevSym, c)
            self._append2stack()
        elif c == self.closingSym:
            if self.prevSym == ":" or (
                self.prevSym == "," and not self.allowCommaBeforeClosingSym
            ):
                raise InvalidCharException(self.prevSym, c)
            if self.itemTmp != "":
                self._append2stack()
            self.tmpContainer = self.type()
            while self.stack[-1] != self.openingSym:
                stackLast = self.stack.pop()
                self._addFn(stackLast)
                if len(self.stack) == 0:
                    raise InbalancedCharException(self.closingSym, self.openingSym)
            self.stack.pop()
            self.stack.append(self.tmpContainer)
        elif c in self.whitespace:
            pass
        else:
            self.itemTmp += c
        if c not in self.whitespace:
            self.prevSym = c

    def _iter(self, limiter="\n"):
        c = True
        while c:
            c = self.file.read(1)
            if c == limiter:
                break
            if c not in self.whitespace and c:
                yield c

    def tocontainer(self, _type, fn=_convert, allowCommaBeforeClosingSym=True):
        if _type == list:
            self.openingSym = "["
            self.closingSym = "]"
        elif _type == set or _type == dict:
            self.openingSym = "{"
            self.closingSym = "}"
        self.type = _type
        self.stack = []
        self.itemTmp = ""
        self.prevSym = " "
        self.startString = None
        self.allowCommaBeforeClosingSym = allowCommaBeforeClosingSym
        self.fn = fn
        for c in self.iter():
            self._processChar(c)
        if self.prevSym != self.closingSym:
            raise StartsEndsException("ends", self.prevSym)
        for elem in self.stack:
            if elem == self.openingSym:
                raise InbalancedCharException(self.openingSym, self.closingSym)
        if self.stack == []:
            raise StrException(
                f"The given string {self} is not a {self.type.__name__} literal"
            )
        return self.stack[0]


class StrException(Exception):
    pass


class InvalidCharException(StrException):
    def __init__(self, prevSym, c) -> None:
        super().__init__(f"Invalid character {prevSym} before {c}")


class InbalancedCharException(StrException):
    def __init__(self, more, less) -> None:
        super().__init__(f"There are more {more} than {less}")


class StartsEndsException(StrException):
    def __init__(self, startsOrEnds, startEndChar) -> None:
        super().__init__(f"The string literal {startsOrEnds} with {startEndChar}")
