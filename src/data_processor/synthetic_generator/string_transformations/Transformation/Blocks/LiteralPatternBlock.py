from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class LiteralPatternBlock(RawPatternBlock):
    NAME = "LITERAL"

    def __init__(self, text=None):
        self._hash = None
        self.text = text

    def update_hash(self):
        self._hash = hash((self.NAME, self.text))


    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
        self.update_hash()

    def unit_apply(self, inp):
        return self.text

    @classmethod
    def extract(cls, inp, blk):
        s = set()
        s.add(LiteralPatternBlock(blk.text))
        return s

    @classmethod
    def get_param_space(cls, inp_lst):
        return {
            'text': inp_lst,
        }

    def is_eq(self, other):
        return self.text == other.text

    def __hash__(self):
        return self._hash

    def str_repr(self):
        return f"LIT:'{self.text}'"
