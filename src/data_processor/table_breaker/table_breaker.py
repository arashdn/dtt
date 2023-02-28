import argparse
import os
import random
import pathlib

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute())


DS_PATH = BASE_PATH + "/data/Datasets/FF_AJ"
OUT_DIR_PATH = BASE_PATH + "/data/Datasets/FF_AJ_Splitted/"

# DS_PATH = BASE_PATH + "/data/Datasets/Single_Substr_10_50"
# OUT_DIR_PATH = BASE_PATH + "/data/Datasets/Single_Substr_10_50_Splitted/"


def get_pairs_from_files(ds_path, tbl_names=[]):
    assert os.path.isdir(ds_path)
    dirs = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]

    res = {}
    res['inputs'] = {}

    for dir in dirs:
        if len(tbl_names) > 0 and dir not in tbl_names:
            continue

        ds_dir = ds_path+'/' + dir
        # assert os.path.exists(ds_dir + "/source.csv")
        # assert os.path.exists(ds_dir + "/target.csv")
        assert os.path.exists(ds_dir + "/rows.txt")
        assert os.path.exists(ds_dir + "/ground truth.csv")

        src_col, target_col = "", ""

        with open(ds_dir + "/rows.txt") as f:
            l = f.readline().strip().split(':')
            src_col = l[0]
            target_col = l[1]
            direction = f.readline().strip()


        pairs = []

        with open(ds_dir + "/ground truth.csv") as f:
            titles = f.readline().strip().split(',')

            if not "source-" + src_col in titles:
                print(ds_dir)

            assert "source-" + src_col in titles
            assert "target-" + target_col in titles

            src_idx = titles.index("source-" + src_col)
            target_idx = titles.index("target-" + target_col)

            if direction.lower() == "target":
                src_idx, target_idx = target_idx, src_idx

            for line in f.readlines():
                items = line.strip().split(',')
                pairs.append((items[src_idx], items[target_idx]))

        res['inputs'][dir] = pairs


    return res


def save_on_file(dir_path, pairs, src_title="stitle", target_title="ttitle"):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


    dir_path = dir_path + "/"

    with open(dir_path + 'rows.txt', 'w') as f:
        print(f"{src_title}:{target_title}", file=f)
        print("source", file=f)

    src_file = open(dir_path + 'source.csv', 'w')
    target_file = open(dir_path + 'target.csv', 'w')
    gt_file = open(dir_path + 'ground truth.csv', 'w')

    print(f"{src_title}", file=src_file)
    print(f"{target_title}", file=target_file)
    print(f"source-{src_title},target-{target_title}", file=gt_file)


    for pair in pairs:
        print(pair[0], file=src_file)
        print(pair[1], file=target_file)
        print(f"{pair[0]},{pair[1]}", file=gt_file)


    src_file.close()
    target_file.close()
    gt_file.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--rel-out-dir', '-o', action='store', type=str, required=False,
                        default=None, help='Relative output directory.')

    parser.add_argument('--rel-dataset-path', '-d', action='store', type=str, required=False,
                        default=None, help='Relative dataset folder path')

    args = parser.parse_args().__dict__

    DS_PATH = BASE_PATH + args['rel_dataset_path'] if args['rel_dataset_path'] is not None else DS_PATH
    OUT_DIR_PATH = BASE_PATH + args['rel_out_dir'] if args['rel_out_dir'] is not None else OUT_DIR_PATH
    OUT_DIR_PATH += "/"

    print(f"Reading dataset: {DS_PATH}")
    print(f"Saving on: {OUT_DIR_PATH}")

    pairs = get_pairs_from_files(DS_PATH, [])

    for table, rows in pairs['inputs'].items():
        print(f"working on {table}")
        random.shuffle(rows)
        train_size = max(2, len(rows)//2)

        pairs_train = rows[:train_size]
        pairs_test = rows[train_size:]

        save_on_file(OUT_DIR_PATH+"smpl_"+table, pairs_train)
        save_on_file(OUT_DIR_PATH+"test_"+table, pairs_test)


