import json
import os
import random
import pathlib

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.absolute())

SAMPLING_RATE = 10

DS_PATH = BASE_PATH + "/data/Datasets/FF_AJ"
OUT_PATH = BASE_PATH + f"/data/SampleSets/ffaj_{SAMPLING_RATE}rt_samples.json"


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


def get_permutations(samples, num_samples, size=-1):
    for i in range(num_samples):

        n = size
        if size == -1:
            p = random.randint(0, 100)
            if p < 50: n = 3
            elif p < 80: n = 4
            else: n = 5

        inp = [random.choice(samples) for j in range(n)]
        out = inp[-1][1]
        inp[-1] = inp[-1][0]

        yield inp, out


def generate_sample_for_data_collection(ds_path, out_path, sampling_rate):

    pairs = get_pairs_from_files(ds_path, [])

    pairs['samples'] = {}

    for key in pairs['inputs']:
        pr = pairs['inputs'][key]
        num_samples = sampling_rate * len(pr)
        ds = list(get_permutations(pr, num_samples, 3))
        pairs['samples'][key] = ds

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)



if __name__ == "__main__":
    generate_sample_for_data_collection(DS_PATH, OUT_PATH, SAMPLING_RATE)


'''
{
   'inputs':
        {
            'dataset1': [(inp1, out1),(inp2, out2),...]
            'dataset2': [(inp1, out1),(inp2, out2),...]
            ...
        },
    'samples':
        {
            'dataset1': [
                            (
                                [  (inp1, out1),(inp2, out2), inp3]  , out3
                            ),
                            ...
                        ]
            'dataset2': 
            ...
        }
}
'''