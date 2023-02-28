import multiprocessing
import os
import pathlib

import basic_generator as bg

from Transformation.Blocks.ReversePatternBlock import ReversePatternBlock
from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
from Transformation.Blocks.ReplacePatternBlock import ReplacePatternBlock

NUM_ROWS = 50
NUMBER_OF_TRANSFORMATIONS = 5
bg.NUMBER_OF_TRANSFORMATIONS = NUMBER_OF_TRANSFORMATIONS
bg.EXAMPLE_SETS_FOR_EACH_TRANSFORMATION = 1
bg.numbers_of_examples_per_set = [NUM_ROWS + 1]

bg.BATCH_SIZE = 5

NAME = "Replace"
LEN = 50
RANGE_OF_INPUT_LEN = (LEN, LEN)
# RANGE_OF_INPUT_LEN = (8, 35)

SINGLE_SET = False

if NAME == "Reverse":

    bg.RANGE_OF_INPUT_LEN = RANGE_OF_INPUT_LEN

    bg.RANGE_OF_UNITS = (1, 1)

    bg.MAX_SUBPATTERN_DEPTH = 0
    bg.PT_BLOCKS = [ReversePatternBlock]
    bg.PT_WEIGHTS = [100]
    SINGLE_SET = True

elif NAME == "Substr":

    bg.RANGE_OF_INPUT_LEN = RANGE_OF_INPUT_LEN

    bg.RANGE_OF_UNITS = (1, 1)

    bg.MAX_SUBPATTERN_DEPTH = 0
    bg.PT_BLOCKS = [PositionPatternBlock]
    bg.PT_WEIGHTS = [100]
    bg.INPUT_CHAR_SET = bg.INPUT_CHAR_SET.replace(" ", "")  # Having only space in the output will lead to empty outputs
    SINGLE_SET = False

elif NAME == "Replace":

    bg.RANGE_OF_INPUT_LEN = RANGE_OF_INPUT_LEN

    bg.RANGE_OF_UNITS = (1, 1)

    bg.MAX_SUBPATTERN_DEPTH = 0
    bg.PT_BLOCKS = [ReplacePatternBlock]
    bg.PT_WEIGHTS = [100]
    SINGLE_SET = False

else:
    raise NotImplementedError


if SINGLE_SET:
    bg.NUMBER_OF_TRANSFORMATIONS = 1
    bg.numbers_of_examples_per_set = [NUM_ROWS * NUMBER_OF_TRANSFORMATIONS + 1]


BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent.absolute())
if RANGE_OF_INPUT_LEN[0] == RANGE_OF_INPUT_LEN[1]:
    DIR_NAME = f"Single_{NAME}_{NUMBER_OF_TRANSFORMATIONS:02}tr_{NUM_ROWS:03}rows_{RANGE_OF_INPUT_LEN[0]:03}len"
else:
    DIR_NAME = f"Single_{NAME}_{NUMBER_OF_TRANSFORMATIONS:02}tr_{NUM_ROWS:03}rows__{RANGE_OF_INPUT_LEN[0]:02}_{RANGE_OF_INPUT_LEN[1]:02}len"
OUT_PATH = BASE_PATH + f"/data/Datasets/{DIR_NAME}/"

bg.INPUT_CHAR_SET = bg.INPUT_CHAR_SET.replace(',', '')

bg.MAX_TIME_LIMIT_PER_PATTERN = 180.0

bg.MULTI_CORE = True
bg.NUM_PROCESSORS = multiprocessing.cpu_count()//2 - 1
bg.NUM_PROCESSORS = multiprocessing.cpu_count() - 2



def save_on_file(dir_path, pairs, tran=""):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    src_title = "src_title"
    target_title = "target_title"

    dir_path = dir_path + "/"

    with open(dir_path + 'rows.txt', 'w') as f:
        print(f"{src_title}:{target_title}", file=f)
        print("source", file=f)
        print(tran, file=f)

    src_file = open(dir_path + 'source.csv', 'w')
    target_file = open(dir_path + 'target.csv', 'w')
    gt_file = open(dir_path + 'ground truth.csv', 'w')

    print(f"{src_title}", file=src_file)
    print(f"{target_title}", file=target_file)
    print(f"source-{src_title},target-{target_title}", file=gt_file)


    for pair in pairs:
        assert "," not in pair[0]
        assert "," not in pair[1]
        print(pair[0], file=src_file)
        print(pair[1], file=target_file)
        print(f"{pair[0]},{pair[1]}", file=gt_file)


    src_file.close()
    target_file.close()
    gt_file.close()


def main():

    res, fails = bg.run()
    while (not SINGLE_SET) and len(res['samples']) != NUMBER_OF_TRANSFORMATIONS:
        print(f"Not enough unique trans({len(res['samples'])}/{NUMBER_OF_TRANSFORMATIONS}). \nretrying the transformation generation...")
        bg.CNT_CUR = multiprocessing.Value('i', 0)
        bg.DONE_BATCHES = multiprocessing.Value('i', 0)
        res, fails = bg.run()

    res = res['samples']

    if SINGLE_SET:
        assert len(res) == 1
        sets = None
        for idx, tr in enumerate(res):
            sets = res[tr][0][0][:-1]
            sets = [sets[i:i + NUM_ROWS] for i in range(0, len(sets), NUM_ROWS)]

        for idx, st in enumerate(sets):
            save_on_file(OUT_PATH+f"{DIR_NAME}_tbl_{idx+1}", st, tr)
    else:
        for idx, tr in enumerate(res):
            pairs = res[tr][0][0][:-1]
            save_on_file(OUT_PATH+f"{DIR_NAME}_tbl_{idx+1}", pairs, tr)


if __name__ == "__main__":
    main()
