from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class ReplacePatternBlock (RawPatternBlock):
    NAME = "REPLACE"


    def __init__(self, old=None, new=None):
        # To init hash:
        self._hash = None
        self._old = None
        self._new = None

        self.old = old
        self.new = new


    def update_hash(self):
        self._hash = hash((self.NAME, self.old, self.new))

    @property
    def old(self):
        return self._old

    @old.setter
    def old(self, old):
        self._old = old
        self.update_hash()

    @property
    def new(self):
        return self._new

    @new.setter
    def new(self, new):
        self._new = new
        self.update_hash()

    def unit_apply(self, inp):
        return inp.replace(self.old, self.new)

    @classmethod
    def extract(cls, inp, blk):
        raise NotImplementedError

    @classmethod
    def get_param_space(cls, inp_lst):
        new_chars = set()
        for inp in inp_lst:
            for c in inp:
                new_chars.add(c)
        return {
            'old': list(new_chars),
            'new': list(new_chars),
        }

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        import random
        new, old = 'a', 'a'
        while new == old:
            old = random.choice(inp_charset)
            new = random.choice(inp_charset)

        return ReplacePatternBlock(old, new)


    def is_eq(self, other):
        return self.new == other.new and self.old == other.old

    def __hash__(self):
        return self._hash
        # return hash((self.splitter, ))

    def str_repr(self):
        return f"Replace: ('{self.old}', {self.new})"

