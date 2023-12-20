import glob
import os
import pathlib

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.absolute())
OUT_DIR_PATH = BASE_PATH + f"/data/output_len8_35/_substr_lens_splitteds_mdl01000"
# OUT_DIR_PATH = BASE_PATH + f"/data/output/_reverse_lens_splitteds_mdl15000"
# OUT_DIR_PATH = BASE_PATH + f"/data/output_noisy_examples_0.0"

FILE_PREFIX = "joinmdl_"


if __name__ == "__main__":

    assert os.path.isdir(OUT_DIR_PATH)
    res = {}

    for file in sorted(glob.glob(OUT_DIR_PATH + f"/{FILE_PREFIX}*.csv")):
        if "______" in file:
            continue

        file_key = os.path.basename(file).replace('.csv', '')
        # file_key = file_key.replace("_Single_Substr_10_50_Splitted", "").replace("_Single_Reverse_10_50_Splitted", "")\
        #     .replace("_FF_AJ_Splitted", "")




        dataset_template = {
            '*': {'p': [], 'r': [], 'f1': [], 'correct': [], 'len': [], 'avg_edit_dist': [], 'avg_norm_edit_dist': [],'time': [], },
                    }

        dataset = dataset_template

        with open(file, "r") as f:
            line = f.readline()
            if not line.strip() == "id,P,R,F1,correct,len,avg_edit_dist,avg_norm_edit_dist,Time":
                raise Exception("Error in file:"+file)

            for line in f.readlines():
                tmp = line.strip().split(",")
                k = "*"

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

    print_ds = ['*']

    print("DS," + "P,R,F1,correct,len,avg_edit_dist,avg_norm_edit_dist,Time," )
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

