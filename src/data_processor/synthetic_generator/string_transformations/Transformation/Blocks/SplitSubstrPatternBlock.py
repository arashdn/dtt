import re

from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class SplitSubstrPatternBlock (RawPatternBlock):
    NAME = "SPLT_SUB"

    def __init__(self, splitter=None, index=None, start=None, end=None):
        # To init hash:
        self._hash = None
        self._splitter = None
        self._index = None
        self._start = None
        self._end = None

        self.splitter = splitter
        self.index = index
        self.start = start
        self.end = end


    def update_hash(self):
        self._hash = hash((self.NAME, self.start, self.end, self.index, self.splitter))

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
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        self._splitter = splitter
        self.update_hash()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.update_hash()

    def unit_apply(self, inp):
        tmp = inp.split(self.splitter)
        try:
            s = tmp[self.index]
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
            'splitter': list(new_chars),
            'index': range(0, 5),
            'start': range(0, len(min(inp_lst, key=len)) - 2),
            'end': range(1, len(min(inp_lst, key=len)) - 1),
        }

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        import random
        indx = random.choice([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3])
        start = random.randint(0, input_max_len//3)
        end = random.randint(start + 1, input_max_len//2)
        return SplitSubstrPatternBlock(random.choice(inp_charset), indx, start, end)

    @classmethod
    def extract(cls, inp, blk):

        tmp = set()
        out = blk.text

        chars = {c for c in inp}
        out_chars = {c for c in out}
        chars = chars - out_chars

        n = len(out)

        for ch in chars:
            spt = inp.split(ch)
            for idx, sp in enumerate(spt):
                if len(sp) > n:  # not >= because it will be same as split
                    matches = [m.start() for m in re.finditer('(?='+re.escape(out)+')', sp)]
                    for m in matches:
                        tmp.add(SplitSubstrPatternBlock(ch, idx, m, m + n))

        return tmp


    def is_eq(self, other):
        return self.splitter == other.splitter and self.index == other.index \
               and self.start == other.start and self.end == other.end

    def __hash__(self):
        return self._hash
        # return hash((self.splitter, self.index, self.start, self.end, ))

    def str_repr(self):
        return f"SplitSubstr: '{self.splitter}', {self.index}, ({self.start}-{self.end})"

# Old Extract: keeps positions from inp instead of splitted text
'''    @classmethod
    def extract(cls, inp, blk):

        tmp = set()
        out = blk.text

        chars = {c for c in inp}
        out_chars = {c for c in out}
        chars = chars - out_chars

        n = len(out)

        for ch in chars:
            indx = [i for i, letter in enumerate(inp) if letter == ch]
            for idx, i in enumerate(indx):

                if idx == 0:
                    l_start = 0
                    l_end = i
                    text = inp[l_start:l_end]
                    if l_end - l_start > n: # not >= because it will be same as split
                        matches = [m.start() for m in re.finditer('(?='+out+')', text)]
                        for m in matches:
                            tmp.add(SplitSubstrPatternBlock(ch, 0, l_start + m, l_start + m + n))

                r_start = i + 1
                r_end = len(inp) if idx + 1 == len(indx) else indx[idx + 1]
                text = inp[r_start:r_end]
                if r_end - r_start > n:  # not >= because it will be same as split
                    matches = [m.start() for m in re.finditer('(?=' + out + ')', text)]
                    for m in matches:
                        tmp.add(SplitSubstrPatternBlock(ch, idx + 1, r_start + m, r_start + m + n))

        return tmp
'''
