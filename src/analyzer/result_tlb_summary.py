import glob
import os
import pathlib

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.absolute())
OUT_DIR_PATH = BASE_PATH + f"/data/output_len8_35/"
# OUT_DIR_PATH = BASE_PATH + f"/data/output_gpt/extend_5_samples/"
# change tmp = file_key.split("_DS_") for GPT3-based models

FILE_PREFIX = "joinmdl_"


if __name__ == "__main__":

    assert os.path.isdir(OUT_DIR_PATH)
    res = {}

    for file in sorted(glob.glob(OUT_DIR_PATH + f"/{FILE_PREFIX}*.csv")):
        if "______" in file:
            continue

        file_key = os.path.basename(file).replace('.csv', '')
        tmp = file_key.split("__DS_")
        # tmp = file_key.split("_DS_")
        assert len(tmp) == 2
        file_key = tmp[0].replace(FILE_PREFIX, "")





        dataset_template = {
            'FF': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'AJ': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'Synthetic_basic': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'Single_Substr': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'Single_Reverse': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'Single_Replace': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
            'names200': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
        }

        dataset = res[file_key] if file_key in res else dataset_template

        with open(file, "r") as f:
            line = f.readline()
            if not line.strip() == "id,P,R,F1,correct,len,avg_edit_dist,avg_norm_edit_dist,Time":
                raise Exception("Error in file:"+file)

            for line in f.readlines():
                tmp = line.strip().split(",")
                k = tmp[0].split('-')[0]
                k = "_".join(k.split("_")[0:2]) if k.startswith("Single_") or k.startswith("Synthetic") else k

                dataset[k]['p'].append(float(tmp[1]))
                dataset[k]['r'].append(float(tmp[2]))
                dataset[k]['f1'].append(float(tmp[3]))
                dataset[k]['correct'].append(float(tmp[4]))
                dataset[k]['len'].append(float(tmp[5]))
                dataset[k]['avg_edit_dist'].append(float(tmp[6]))
                dataset[k]['avg_norm_edit_dist'].append(float(tmp[7]))
                dataset[k]['time'].append(float(tmp[8]))

            res[file_key] = dataset


    # print(res)

    fp = None

    print_ds = ['AJ', 'FF', 'Synthetic_basic', 'Single_Substr', 'Single_Replace', 'Single_Reverse']

    print("," + ",,,,,,,,".join(print_ds))
    s = "DS,"
    for ds in print_ds:
        s += "P,R,F1,correct,len,avg_edit_dist,avg_norm_edit_dist,Time,"
    print(s, file=fp)
    for model, ds in res.items():
        print(model + ',', file=fp, end='')
        for pds in print_ds:

            is_ok = len(ds[pds]['p']) != 0

            avg_p = sum(ds[pds]['p']) / len(ds[pds]['p']) if is_ok else "-"
            avg_r = sum(ds[pds]['r']) / len(ds[pds]['r']) if is_ok else "-"
            avg_f1 = sum(ds[pds]['f1']) / len(ds[pds]['f1']) if is_ok else "-"
            avg_correct = sum(ds[pds]['correct']) / len(ds[pds]['correct']) if is_ok else "-"
            avg_len = sum(ds[pds]['len']) / len(ds[pds]['len']) if is_ok else "-"
            avg_avg_edit_dist = sum(ds[pds]['avg_edit_dist']) / len(ds[pds]['avg_edit_dist']) if is_ok else "-"
            avg_avg_norm_edit_dist = sum(ds[pds]['avg_norm_edit_dist']) / len(ds[pds]['avg_norm_edit_dist']) if is_ok else "-"
            avg_time = sum(ds[pds]['time']) / len(ds[pds]['time']) if is_ok else "-"

            print(f"{avg_p},{avg_r},{avg_f1},{avg_correct},{avg_len},"+
                  f"{avg_avg_edit_dist},{avg_avg_norm_edit_dist},{avg_time},", file=fp, end='')

        print(file=fp)

