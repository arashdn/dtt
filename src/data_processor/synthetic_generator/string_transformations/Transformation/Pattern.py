from itertools import chain

from Transformation.Blocks.RawPatternBlock import RawPatternBlock


class BasicPatternBlock:
    TYPE_TOKEN = 1
    TYPE_STR = 2
    TYPES = [TYPE_TOKEN, TYPE_STR]

    def __init__(self, text, type, start=None, end=None, begin_sep=None, end_sep=None):
        self.text = text
        self.type = type
        self.start = start
        self.end = end
        self.begin_sep = begin_sep
        self.end_sep = end_sep

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in BasicPatternBlock.TYPES:
            raise ValueError(f"Type {type} is not accepted.")
        self._type = type

    def __eq__(self, other):
        return self.type == other.type and self.text == other.text and self.start == other.start \
               and self.end == other.end and self.begin_sep == other.begin_sep and self.end_sep == other.end_sep

    def __hash__(self):
        return hash((self.type, self.text, self.start, self.end, self.begin_sep, self.end_sep))


class BasicPattern:
    def __init__(self, inp, goal, blocks=[]):
        self.inp = inp
        self.goal = goal
        self.blocks = blocks

    @property
    def inp(self):
        return self._inp

    @inp.setter
    def inp(self, inp):
        self._inp = inp

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        self._goal = goal

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, blocks):
        if type(blocks) is not list:
            raise ValueError('blocks must be a list')
        for blk in blocks:
            if type(blk) is not BasicPatternBlock:
                raise ValueError('each block must be PatternBlock')
        self._blocks = blocks

    def replace(self, index, blocks):
        if type(blocks) is not list:
            raise ValueError('blocks must be a list')
        if index < 0 or index > len(self) - 1:
            raise ValueError('Incorrect Index')
        for blk in blocks:
            if type(blk) is not BasicPatternBlock:
                raise ValueError('each block must be PatternBlock')

        self._blocks = list(chain(self._blocks[:index], blocks, self._blocks[index + 1:]))

    @property
    def coverage(self):
        return sum(len(b.text) for b in self.blocks if b.type == BasicPatternBlock.TYPE_TOKEN)/len(self.goal)
    # @TODO: consider number of tokens

    @property
    def token_num_score(self):
        num = sum(1 for b in self.blocks if b.type == BasicPatternBlock.TYPE_TOKEN)
        if num == 0:
            return 0
        return 1/num

    @property
    def score(self):
        alpha = 0.95
        return (alpha * self.coverage) + (1 - alpha) * self.token_num_score

    @property
    def num_tokens(self):
        return sum(1 for b in self.blocks if b.type == BasicPatternBlock.TYPE_TOKEN)

    @staticmethod
    def blocks_merge_literals(pt_blks):
        blks = []
        txt = ""
        for bl in pt_blks:
            if bl.type == BasicPatternBlock.TYPE_STR:
                txt += bl.text
            else:
                if txt != "":
                    blks.append(BasicPatternBlock(txt, BasicPatternBlock.TYPE_STR))
                    txt = ""
                blks.append(bl)
        if txt != "":
            blks.append(BasicPatternBlock(txt, BasicPatternBlock.TYPE_STR))

        return blks

    @staticmethod
    def merge_literals(pt):
        return BasicPattern(pt.inp, pt.goal, BasicPattern.blocks_merge_literals(pt.blocks))

    def __str__(self):
        s = "{"
        for b in self.blocks:
            if b.type == BasicPatternBlock.TYPE_STR:
                s += f"[str:{b.text}], "
            elif b.type == BasicPatternBlock.TYPE_TOKEN:
                s += f"[tok:{b.text}-({b.start}-{b.end})-({b.begin_sep}:{b.end_sep})], "
        return s+"}"

    def __len__(self):
        return len(self.blocks)

    def __hash__(self):
        return hash(101*len(self.blocks))

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for i in range(len(self.blocks)):
            if self.blocks[i] != other.blocks[i]:
                return False
        return True

    def __lt__(self, other):
        return self.score < other.score


class Pattern:
    def __init__(self, blocks=[]):
        self._hash = None
        self.blocks = blocks


    def update_hash(self):
        self._hash = hash(tuple([b._hash for b in self.blocks]))
        # self._hash = hash(101*len(self.blocks))

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, inpt):
        if type(inpt) not in [list, tuple]:
            raise ValueError('blocks must be a list')

        blocks = []
        for inp in inpt:
            if type(inp) is not tuple:
                if type(inp).mro()[1] != RawPatternBlock:   # is a subclass of RawPatternBlock
                    raise ValueError('input type not valid')
                blocks.append(inp)
            else:
                for i in inp:
                    if type(i).mro()[1] != RawPatternBlock:  # is a subclass of RawPatternBlock
                        raise ValueError('input type not valid')
                    blocks.append(i)

        self._blocks = blocks
        self.update_hash()

    def apply(self, inp):
        s = ""
        for b in self.blocks:
            try:
                s += b.apply(inp)
            except IndexError:
                return None
        return s


    def fast_apply(self, inp, out, wrongs=set(), pattern_counts=[0, 0, 0]):
        for b in self.blocks:
            if b in wrongs:
                pattern_counts[2] += 1
                return None

        s = ""
        for b in self.blocks:
            try:
                ot = b.apply(inp)
                if ot not in out:
                    wrongs.add(b)
                    return None
                s += ot
            except IndexError:
                wrongs.add(b)
                return None
        return s

    def apply_get_single_diff_block(self, inp, out):
        s = ""
        diff_block = -1
        diff_text = None
        diff_start_idx = None
        new_s = ""

        for idx, b in enumerate(self.blocks):
            try:
                new = b.apply(inp)
            except IndexError:
                return None, None
            s += new
            if diff_block == -1:  # no different block so far
                if not (len(s) <= len(out) and s == out[0:len(s)]):
                    diff_block = idx
                    diff_text = new
                    diff_start_idx = len(s) - len(new)
            else:
                new_s += new
        if diff_block == -1 and not(len(s) < len(out) and s == out[0:len(s)]):
            return s, None
        elif diff_block == -1 and len(s) < len(out) and s == out[0:len(s)]:
            return s, {
                'diff_block': len(self.blocks) - 1,
                'diff_text': '',
                'diff_out': out[len(s):],
                'diff_start_idx': len(s),
                'diff_end_idx': len(out),
            }
        elif len(out) - diff_start_idx - len(new_s) >= 0 and new_s == out[-len(new_s):]:
            return s, {
                'diff_block': diff_block,
                'diff_text': diff_text,
                'diff_out': out[diff_start_idx:len(out) - len(new_s)],
                'diff_start_idx': diff_start_idx,
                'diff_end_idx': len(out) - len(new_s),
            }
        else:
            return s, None

    def __repr__(self):
        s = "{"
        for b in self.blocks:
            s += str(b)
        return s+"}"

    def __len__(self):
        return len(self.blocks)

    def __hash__(self):
        return self._hash
        # return hash(101*len(self.blocks))

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for i in range(len(self.blocks)):
            if type(self.blocks[i]) != type(other.blocks[i]):
                return False
            # from pattern.Patterns.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock
            # if type(self.blocks[i]) == SplitSubstrPatternBlock:
            #     return False
            #     continue
            if self.blocks[i] != other.blocks[i]:
                return False
        return True

    # def __lt__(self, other):
    #     from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
    #     from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
    #     from Transformation.Blocks.TokenPatternBlock import TokenPatternBlock
    #     from Transformation.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock
    #     from Transformation.Blocks.TwoCharSplitSubstrPatternBlock import TwoCharSplitSubstrPatternBlock
    #     from Transformation.Blocks.SplitSplitSubstrPatternBlock import SplitSplitSubstrPatternBlock
    #     BLKS_MAP = {
    #         LiteralPatternBlock: 6,
    #         PositionPatternBlock: 2,
    #         TokenPatternBlock: 1,
    #         SplitSubstrPatternBlock: 3,
    #         TwoCharSplitSubstrPatternBlock: 4,
    #         SplitSplitSubstrPatternBlock: 5
    #     }
    #     num1 = 0
    #     for blk in self.blocks:
    #         assert type(blk) in BLKS_MAP
    #         num1 *= 10
    #         num1 += BLKS_MAP[type(blk)]
    #
    #     num2 = 0
    #     for blk in other.blocks:
    #         assert type(blk) in BLKS_MAP
    #         num2 *= 10
    #         num2 += BLKS_MAP[type(blk)]
    #
    #
    #     if num1 != num2:
    #         return num1 < num2
    #     else:
    #         # @TODO: to be added
    #         return True
    #
