from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class PositionPatternBlock (RawPatternBlock):
    NAME = "POS"

    def __init__(self, start=None, end=None):
        # To init hash:
        self._hash = None
        self._start = None
        self._end = None

        self.start = start
        self.end = end

    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end))

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

    def unit_apply(self, inp):
        if self.end > len(inp):
            raise IndexError("end > len(input)")
        return inp[self.start:self.end]

    @classmethod
    def extract(cls, inp, blk):
        s = set()
        s.add(PositionPatternBlock(blk.start, blk.end))
        return s

    @classmethod
    def get_param_space(cls, inp_lst):
        return {
                   'start': range(0, len(min(inp_lst, key=len)) - 1),
                   'end': range(1, len(min(inp_lst, key=len))),
               }

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        import random
        start = random.randint(0, input_max_len//2)
        end = random.randint(start + 1, input_max_len)
        return PositionPatternBlock(start, end)

    def is_eq(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash
        # return hash((self.start, self.end,))

    def str_repr(self):
        return f"Substr:({self.start}-{self.end})"
