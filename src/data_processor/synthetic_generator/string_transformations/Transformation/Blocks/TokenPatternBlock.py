from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class TokenPatternBlock (RawPatternBlock):
    NAME = "TOK"

    def __init__(self, splitter=None, index=None):
        # To init hash:
        self._hash = None
        self._splitter = None
        self._index = None

        self.splitter = splitter
        self.index = index


    def update_hash(self):
        if type(self.index) == list:
            return 1
        else:
            self._hash = hash((self.NAME, self.index, self.splitter))

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
        s = ""
        tmp = inp.split(self.splitter)
        try:
            if type(self.index) == list:
                for ind in self.index:
                    s += tmp[ind] + self.splitter
                s = s[:-1]  # remove last extra splitter
            else:
                s += tmp[self.index]
        except IndexError:
            raise IndexError("Split index not in the array")
        return s

    @classmethod
    def extract(cls, inp, blk):
        tmp = set()
        sp = blk.end_sep
        if sp is not None:
            parts = inp.split(sp)
            if sp not in blk.text:
                for idx, part in enumerate(parts):
                    if part == blk.text:
                        tmp.add(TokenPatternBlock(sp, idx))
                        # if ARRAY_MINUS:
                        #     m = -(idx + 1)
                        #     assert parts[idx] == part == parts[m]
                        #     tmp.add(TokenPatternBlock(sp, m))
            else:
                txts = blk.text.split(sp)
                n = len(txts)
                for idx, part in enumerate(parts):
                    if idx <= len(parts) - n:
                        is_ok = True
                        ttt = []
                        for i, txt in enumerate(txts):
                            ttt.append(idx + i)
                            if txt != parts[idx + i] or txt == '':
                                is_ok = False
                                break
                        if is_ok:
                            # Just verify #
                            s = ""
                            for i in ttt:
                                s += parts[i] + sp
                            s = s[:-1]
                            assert s == blk.text
                            tmp.add(TokenPatternBlock(sp, ttt))

        sp = blk.begin_sep
        if sp is not None:
            parts = inp.split(sp)
            if sp not in blk.text:
                for idx, part in enumerate(parts):
                    if part == blk.text:
                        tmp.add(TokenPatternBlock(sp, idx))
                        # if ARRAY_MINUS:
                        #     m = -(idx + 1)
                        #     assert parts[idx] == part == parts[m]
                        #     tmp.add(TokenPatternBlock(sp, m))
            else:
                txts = blk.text.split(sp)
                n = len(txts)
                for idx, part in enumerate(parts):
                    if idx <= len(parts) - n:
                        is_ok = True
                        ttt = []
                        for i, txt in enumerate(txts):
                            ttt.append(idx + i)
                            if txt != parts[idx + i] or txt == '':
                                is_ok = False
                                break
                        if is_ok:
                            # Just verify #
                            s = ""
                            for i in ttt:
                                s += parts[i] + sp
                            s = s[:-1]
                            assert s == blk.text
                            tmp.add(TokenPatternBlock(sp, ttt))

        if blk.start > 0:
            sp = blk.text[0]
            parts = inp.split(sp)
            for idx, part in enumerate(parts):
                if sp + part == blk.text:
                    from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
                    tmp.add(
                        (LiteralPatternBlock(sp), TokenPatternBlock(sp, idx))
                    )
                    # if ARRAY_MINUS:
                    #     m = -(idx + 1)
                    #     assert parts[idx] == part == parts[m]
                    #     tmp.add(
                    #         (LiteralPatternBlock(sp), TokenPatternBlock(sp, m))
                    #     )

        return tmp

    @classmethod
    def get_param_space(cls, inp_lst):
        new_chars = set()
        for inp in inp_lst:
            for c in inp:
                new_chars.add(c)
        return {
            'splitter': list(new_chars),
            'index': range(0, 5),
        }

    @classmethod
    def get_random(cls, inp_charset, input_max_len):
        import random
        return TokenPatternBlock(random.choice(inp_charset), random.choice([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3]))

    def is_eq(self, other):
        if self.splitter != other.splitter:
            return False
        if type(self.index) is list:
            if type(other.index) is list:
                return len(self.index) == len(other.index) \
                       and len(self.index) == sum([1 for i, j in zip(self.index, other.index) if i == j])
            else:
                return False
        else:
            return self.index == other.index

    def __hash__(self):
        return self._hash
        # return hash((self.splitter, ))

    def str_repr(self):
        return f"Split: '{self.splitter}', {self.index}"

