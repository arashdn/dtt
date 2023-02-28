from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class SplitSplitSubstrPatternBlock (RawPatternBlock):
    NAME = "SPLT_SPLT_SUB"

    def __init__(self, splitter1=None, index1=None, splitter2=None, index2=None, start=None, end=None):
        # To init hash:
        self._hash = None
        self._splitter1 = None
        self._index1 = None
        self._splitter2 = None
        self._index2 = None
        self._start = None
        self._end = None

        self.splitter1 = splitter1
        self.index1 = index1
        self.splitter2 = splitter2
        self.index2 = index2
        self.start = start
        self.end = end


    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end, self.index1, self.splitter1, self.index2, self.splitter2))

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start
        self.update_hash()

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end
        self.update_hash()

    @property
    def splitter1(self):
        return self._splitter1

    @splitter1.setter
    def splitter1(self, splitter1):
        self._splitter1 = splitter1
        self.update_hash()

    @property
    def index1(self):
        return self._index1

    @index1.setter
    def index1(self, index1):
        self._index1 = index1
        self.update_hash()

    @property
    def splitter2(self):
        return self._splitter2

    @splitter2.setter
    def splitter2(self, splitter2):
        self._splitter2 = splitter2
        self.update_hash()

    @property
    def index2(self):
        return self._index2

    @index2.setter
    def index2(self, index2):
        self._index2 = index2
        self.update_hash()

    def unit_apply(self, inp):
        tmp = inp.split(self.splitter1)
        try:
            s = tmp[self.index1]
            tmp2 = s.split(self.splitter2)
            r = tmp2[self.index2]
            end = self.end
            if self.end > len(r):
                end = len(r)
                # raise ValueError("end > len(input)")
            return r[self.start:end]
        except IndexError:
            raise IndexError("Split index not in the array")

    @classmethod
    def get_param_space(cls, inp_lst):
        new_chars = set()
        for inp in inp_lst:
            for c in inp:
                new_chars.add(c)
        return {
            'splitter1': list(new_chars),
            'index1': range(0, 4),
            'splitter2': list(new_chars),
            'index2': range(0, 4),
            'start': range(0, len(min(inp_lst, key=len)) - 3),
            'end': range(1, len(min(inp_lst, key=len)) - 2),
        }

    def is_eq(self, other):
        return self.splitter == other.splitter and self.index == other.index \
               and self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash
        # return hash((self.splitter, self.index, self.start, self.end, ))

    def str_repr(self):
        return f"SplitSubstr: '{self.splitter}', {self.index}, ({self.start}-{self.end})"

