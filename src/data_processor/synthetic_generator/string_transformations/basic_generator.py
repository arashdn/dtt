# @TODO: Transformation based on the input formatting (conditional), also solves the issue of not accurate parameter selection
# @TODO: use log normal dist. for length
# @TODO: use punctuation in parameters more
# maybe have some input formats???

import glob
import json
import multiprocessing
import os
import pathlib
import random
import sys
import time
# import sys
# TR_LIB_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.absolute())+"/Column_Matcher/codes/src/"
# sys.path += [TR_LIB_PATH]
from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
from Transformation.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock
from Transformation.Blocks.TokenPatternBlock import TokenPatternBlock
from Transformation.Blocks.LowerPatternBlock import LowerPatternBlock
from Transformation.Blocks.UpperPatternBlock import UpperPatternBlock
from Transformation.Pattern import Pattern


NUMBER_OF_TRANSFORMATIONS = 15000
RANGE_OF_UNITS = (3, 6)
EXAMPLE_SETS_FOR_EACH_TRANSFORMATION = 10
# RANGE_OF_INPUT_LEN = (8, 35)
RANGE_OF_INPUT_LEN = (5, 60)

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent.absolute())
OUT_PATH = BASE_PATH + f"/data/SampleSets/basic_synth_{NUMBER_OF_TRANSFORMATIONS:05}_{EXAMPLE_SETS_FOR_EACH_TRANSFORMATION}.json"


MAX_TIME_LIMIT_PER_PATTERN = 180.0

MULTI_CORE = True
NUM_PROCESSORS = multiprocessing.cpu_count()//2 - 1
NUM_PROCESSORS = multiprocessing.cpu_count() - 2
BATCH_SIZE = 100
CNT_CUR = multiprocessing.Value('i', 0)
DONE_BATCHES = multiprocessing.Value('i', 0)


LETTERS = 'abcdefghijklmnopqrstuvwxyz1234567890'
CAP_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHA_NUMERICS = LETTERS + CAP_LETTERS
SYMBOLS = '   -,.?()*&'  # more chance for space
INPUT_CHAR_SET = LETTERS + CAP_LETTERS + SYMBOLS


# PT_BLOCKS = [LiteralPatternBlock,LiteralPatternBlock,PositionPatternBlock,PositionPatternBlock,TokenPatternBlock,TokenPatternBlock,SplitSubstrPatternBlock]
# PT_BLOCKS = [LiteralPatternBlock, LiteralPatternBlock, LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock, SplitSubstrPatternBlock]
PT_BLOCKS = [LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock]  # SplitSubstrPatternBlock
PT_WEIGHTS = [50, 20, 30]
MAX_SUBPATTERN_DEPTH = 1


SUB_PATTERNS = {
    PositionPatternBlock.NAME: {
        'subpatterns': [None, UpperPatternBlock, LowerPatternBlock],
        'weights': [60, 20, 20]
    },
    TokenPatternBlock.NAME: {
        'subpatterns': [None, PositionPatternBlock, TokenPatternBlock, UpperPatternBlock, LowerPatternBlock],
        'weights': [45, 30, 15, 5, 5]
    },
    LowerPatternBlock.NAME: {
        'subpatterns': [None],
        'weights': [100]
    },
    UpperPatternBlock.NAME: {
        'subpatterns': [None],
        'weights': [100]
    },
}


# numbers_of_examples_per_set = [3,3,3,3,3,4,4,4,4,5,5,5,6,6,7,7,8]
numbers_of_examples_per_set = [3]


def get_subpatterns(block, depth):
    if depth > MAX_SUBPATTERN_DEPTH:
        return None

    tmp = SUB_PATTERNS[block.NAME]
    clss = random.choices(tmp['subpatterns'], weights=tmp['weights'], k=1)[0]
    if clss is None:
        return None
    sub = clss.get_random(INPUT_CHAR_SET, RANGE_OF_INPUT_LEN[1] // 2)
    sub.subpattern = get_subpatterns(sub, depth + 1)
    return sub

def generate_input(len_range):
    inp_len = random.randint(len_range[0], len_range[1])
    return random.choice(LETTERS) + ''.join(random.choice(INPUT_CHAR_SET) for i in range(inp_len - 2)) + random.choice(LETTERS)


def merge_literals(pt):
    blks = []
    txt = ""
    for bl in pt.blocks:
        if type(bl) == LiteralPatternBlock:
            txt += bl.text
        else:
            if txt != "":
                blks.append(LiteralPatternBlock(txt))
                txt = ""
            blks.append(bl)
    if txt != "":
        blks.append(LiteralPatternBlock(txt))

    return Pattern(blks)


def get_sets_for_transformation(transformation, num_sets, range_of_inp_len):
    all_sets = []

    for i in range(num_sets):
        size = random.choice(numbers_of_examples_per_set)
        res = []
        for j in range(size):
            src = generate_input(range_of_inp_len)
            target = transformation.apply(src)

            start_time = time.time()
            while target is None:
                src = generate_input(range_of_inp_len)
                target = transformation.apply(src)
                if time.time() - start_time > MAX_TIME_LIMIT_PER_PATTERN:
                    return None


            res.append((src, target))

        all_sets.append(res)

    return all_sets


def generate_samples(number, batch_num=-1, total_batches=-1):
    # res = {
    #     'inputs': {},
    #     'samples': {},
    # }

    samples = {}

    global CNT_CUR


    fails = []
    i = 0
    while i < number:
        unit_num = random.randint(RANGE_OF_UNITS[0], RANGE_OF_UNITS[1])
        blks = []
        for j in range(unit_num):
            cls = random.choices(PT_BLOCKS, weights=PT_WEIGHTS, k=1)[0]
            pt = None
            if cls is LiteralPatternBlock:
                lit_len = random.randint(1, RANGE_OF_INPUT_LEN[1] // 4)
                pt = LiteralPatternBlock(''.join(random.choice(INPUT_CHAR_SET) for i in range(lit_len)))
            else:
                pt = cls.get_random(INPUT_CHAR_SET, RANGE_OF_INPUT_LEN[1])
                pt.subpattern = get_subpatterns(pt, 1)

            blks.append(pt)

        tr = merge_literals(Pattern(blks))
        # print(tr)
        sets = get_sets_for_transformation(tr, EXAMPLE_SETS_FOR_EACH_TRANSFORMATION, RANGE_OF_INPUT_LEN)

        if sets is None:
            print(f"Batch {batch_num}/{total_batches} -- {i}/{number}({CNT_CUR.value}/{NUMBER_OF_TRANSFORMATIONS}):{tr}" +
                  f"\n   Failed to find in {MAX_TIME_LIMIT_PER_PATTERN} sec. skipping")
            fails.append(tr)
            continue


        if str(tr) not in samples:
            samples[str(tr)] = []

        for st in sets:
            out = st[-1][1]
            st[-1] = st[-1][0]
            samples[str(tr)].append((st, out))


        with CNT_CUR.get_lock():
            CNT_CUR.value += 1
        i += 1
        print(f"Batch {batch_num}: ({DONE_BATCHES.value}/{total_batches}) -- {i}/{number} ({CNT_CUR.value}/{NUMBER_OF_TRANSFORMATIONS}):{tr}")

    with DONE_BATCHES.get_lock():
        DONE_BATCHES.value += 1
    return samples, fails


def run():

    num_batchs = NUMBER_OF_TRANSFORMATIONS // BATCH_SIZE
    batches = [BATCH_SIZE for b in range(num_batchs)]
    if NUMBER_OF_TRANSFORMATIONS % BATCH_SIZE != 0:
        batches += [NUMBER_OF_TRANSFORMATIONS % BATCH_SIZE]

    all_samples = {}
    all_fails = []

    params = []

    for i, batch in enumerate(batches):
        if MULTI_CORE:
            params.append((batch, i + 1, len(batches)))
        else:
            samples, fails = generate_samples(batch, i + 1, len(batches))
            all_samples.update(samples)
            all_fails += fails

    if MULTI_CORE:
        print(f"Using {NUM_PROCESSORS} processes...")

        # @TODO: Support spawn
        if sys.platform in ('win32', 'msys', 'cygwin'):
            print("fork based multi core processing works only on *NIX type operating systems.")
            sys.exit(1)

        from multiprocessing import get_context
        pool = get_context('fork').Pool(processes=NUM_PROCESSORS)
        # pool = multiprocessing.Pool(processes=NUM_PROCESSORS)

        rets = pool.starmap(generate_samples, params)
        pool.close()

        pool.join()

        for ret in rets:
            samples, fails = ret
            all_samples.update(samples)
            all_fails += fails

    res = {
        'inputs': {},
        'samples': all_samples,
    }

    return res, all_fails


def main():
    res, fails = run()

    print("Failed Trans:")
    for f in fails:
        print("   " + str(f))

    print(f"total of {len(res['samples'])} transformations, {len(fails)} fails.")
    print("writing result file...")
    print(OUT_PATH)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()





