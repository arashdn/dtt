
class RawPatternBlock:
    NAME = "PATTERN"

    _subpattern = None

    @property
    def subpattern(self):
        return self._subpattern

    @subpattern.setter
    def subpattern(self, subpattern):
        self._subpattern = subpattern


    def __init__(self, ):
        pass

    def unit_apply(self, inp):
        raise NotImplementedError


    def apply(self, inp):
        if self.subpattern is None:
            return self.unit_apply(inp)
        else:
            cur = self.unit_apply(inp)
            return self.subpattern.apply(cur)


    @classmethod
    def extract(cls, inp, blk):
        return set()

    def is_eq(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        if self.subpattern is None:
            if other.subpattern is None:
                self.is_eq(other)
            return False

        else:
            if other.subpattern is None:
                return False
            return self.is_eq(other) and self.subpattern == other.subpattern


    def __hash__(self):
        raise NotImplementedError


    def str_repr(self):
        return "Unknown pattern representation"

    def __repr__(self):
        if self.subpattern is None:
            return f"[{self.str_repr()} ], "
        else:
            return f"[{self.str_repr()} -> {str(self.subpattern)} ], "
