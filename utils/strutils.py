class Str(str):
    @staticmethod
    def _convert(item: str):
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
                        return item

    def _initElementalContainer(self):
        self.tmpContainer = self.type()

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

    def _append2stack(self, fn):
        self.stack.append(fn(self.itemTmp))
        self.itemTmp = ""

    def tolist(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(list, fn, allowCommaBeforeClosingSym)

    def toset(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(set, fn, allowCommaBeforeClosingSym)

    def todict(self, fn=_convert, allowCommaBeforeClosingSym=True):
        return self.tocontainer(dict, fn, allowCommaBeforeClosingSym)

    def tocontainer(self, _type, fn, allowCommaBeforeClosingSym):
        if _type == list:
            openingSym = "["
            closingSym = "]"
        elif _type == set or _type == dict:
            openingSym = "{"
            closingSym = "}"
        self.type = _type
        self.stack = []
        self.itemTmp = ""
        prevSym = " "
        startString = None
        for c in self:
            if startString == None and c != " ":
                startString = c
                if startString != openingSym:
                    raise StartsEndsException("starts", startString)

            if c == openingSym:
                if prevSym not in (" ", ",", openingSym) and (
                    self.type != dict or prevSym != ":"
                ):
                    raise InvalidCharException(prevSym, c)
                self.itemTmp = ""
                self.stack.append(openingSym)
            elif c == ",":
                if prevSym in (",", ":", openingSym):
                    raise InvalidCharException(prevSym, c)
                if self.itemTmp != "":
                    self._append2stack(fn)
            elif c == ":":
                if prevSym in (",", ":", openingSym, closingSym):
                    raise InvalidCharException(prevSym, c)
                self._append2stack(fn)
            elif c == closingSym:
                if prevSym == ":" or (
                    prevSym == "," and not allowCommaBeforeClosingSym
                ):
                    raise InvalidCharException(prevSym, c)
                if self.itemTmp != "":
                    self._append2stack(fn)
                self._initElementalContainer()
                while self.stack[-1] != openingSym:
                    stackLast = self.stack.pop()
                    self._addFn(stackLast)
                    if len(self.stack) == 0:
                        raise InbalancedCharException(closingSym, openingSym)
                self.stack.pop()
                self.stack.append(self.tmpContainer)
            else:
                self.itemTmp += c
            if c != " ":
                prevSym = c
        if prevSym != closingSym:
            raise StartsEndsException("ends", prevSym)
        for elem in self.stack:
            if elem == openingSym:
                raise InbalancedCharException(openingSym, closingSym)
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
