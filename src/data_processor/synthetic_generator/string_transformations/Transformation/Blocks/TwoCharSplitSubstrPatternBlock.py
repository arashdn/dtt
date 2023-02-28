import re

from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class TwoCharSplitSubstrPatternBlock (RawPatternBlock):
    NAME = "TWO_CH_SPLT_SUB"

    def __init__(self, char1=None, char2=None, index=None, start=None, end=None):
        # To init hash:
        self._hash = None
        self._char1 = None
        self._char2 = None
        self._index = None
        self._start = None
        self._end = None

        self.char1 = char1
        self.char2 = char2
        self.index = index
        self.start = start
        self.end = end


    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end, self.index, self.char1, self.char2))


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
    def char1(self):
        return self._char1

    @char1.setter
    def char1(self, char1):
        self._char1 = char1
        self.update_hash()

    @property
    def char2(self):
        return self._char2

    @char2.setter
    def char2(self, char2):
        self._char2 = char2
        self.update_hash()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.update_hash()

    def unit_apply(self, inp):
        res = re.findall(re.escape(self.char1) + '(.*?)' + re.escape(self.char2), inp)

        try:
            s = res[self.index]
            end = self.end
            if self.end > len(s):
                end = len(s)
                # raise ValueError("end > len(input)")
            return s[self.start:end]
        except IndexError:
            raise IndexError("Split index not in the array")

    @classmethod
    def get_param_space(cls, inp_lst):
        new_chars = set()
        for inp in inp_lst:
            for c in inp:
                new_chars.add(c)
        return {
            'char1': list(new_chars),
            'char2': list(new_chars),
            'index': range(0, 5),
            'start': range(0, len(min(inp_lst, key=len)) - 2),
            'end': range(1, len(min(inp_lst, key=len)) - 1),
        }

    @classmethod
    def extract(cls, inp, blk):

        if blk.start == 0 or blk.end == len(inp):
            return set()

        tmp = set()
        out = blk.text

        chars1 = {c for c in inp[0:blk.start]}
        chars2 = {c for c in inp[blk.end:]}
        out_chars = {c for c in out}
        chars1 = chars1 - out_chars
        chars2 = chars2 - out_chars

        if len(chars1) == 0 or len(chars2) == 0:
            return set()


        n = len(out)

        for c1 in chars1:
            for c2 in chars2:
                if c1 != c2:
                    spt = re.findall(re.escape(c1) + '(.*?)' + re.escape(c2), inp)
                    for idx, sp in enumerate(spt):
                        if len(sp) >= n:
                            matches = [m.start() for m in re.finditer('(?=' + re.escape(out) + ')', sp)]
                            for m in matches:
                                tmp.add(TwoCharSplitSubstrPatternBlock(c1, c2, idx, m, m + n))

        return tmp


    def is_eq(self, other):
        return self.char1 == other.char1 and self.char2 == other.char2 and self.index == other.index \
               and self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash
        # return hash((self.char1, self.char2, self.index, self.start, self.end, ))

    def str_repr(self):
        return f"[TwoCharSplitSubstr: ('{self.char1}','{self.char2}'), {self.index}, ({self.start}-{self.end})"

