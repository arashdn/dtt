import multiprocessing
import os
import pathlib

import basic_generator as bg

NUM_ROWS = 100
bg.NUMBER_OF_TRANSFORMATIONS = 10
bg.RANGE_OF_UNITS = (3, 6)
bg.EXAMPLE_SETS_FOR_EACH_TRANSFORMATION = 1
bg.numbers_of_examples_per_set = [NUM_ROWS + 1]

bg.BATCH_SIZE = 5

LEN1 = 8
LEN2 = 35

bg.RANGE_OF_INPUT_LEN = (LEN1, LEN2)

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent.absolute())
DIR_NAME = f"Synthetic_basic_{bg.NUMBER_OF_TRANSFORMATIONS:02}tr_{NUM_ROWS:03}rows__{LEN1:02}_{LEN2:02}len"
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
    res = res['samples']


    for idx, tr in enumerate(res):
        pairs = res[tr][0][0][:-1]
        save_on_file(OUT_PATH+f"{DIR_NAME}_tbl_{idx+1}", pairs, tr)


if __name__ == "__main__":
    main()
