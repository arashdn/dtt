from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class LowerPatternBlock(RawPatternBlock):
    NAME = "LOWER"

    def __init__(self, text=None):
        self._hash = None

    def update_hash(self):
        self._hash = hash(self.NAME)


    def unit_apply(self, inp):
        return inp.lower()

    @classmethod
    def extract(cls, inp, blk):
        s = set()
        s.add(LowerPatternBlock())
        return s

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        return LowerPatternBlock()

    @classmethod
    def get_param_space(cls, inp_lst):
        return {

        }

    def is_eq(self, other):
        return True

    def __hash__(self):
        return self._hash

    def str_repr(self):
        return f"Lower()"
