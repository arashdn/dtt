import json
import random
import pandas as pd


TR_TOKEN = "</tr>"
EOE_TOKEN = "</eoe>"



def src_prepare(inp_pairs):
    res = ""
    assert len(inp_pairs) > 1

    # reading examples with the target
    for pair in inp_pairs[:-1]:
        assert len(pair) == 2
        res += pair[0] + TR_TOKEN + pair[1] + EOE_TOKEN

    # last example without target
    assert type(inp_pairs[-1]) == str
    res += inp_pairs[-1] + TR_TOKEN
    return res


def get_dataframe(samples_path, shrink_samples=0):
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

    with open(samples_path) as f:
        data = json.load(f)

    samples = []

    for dataset in data['samples']:
        for sample in data['samples'][dataset]:
            value = {
                'sample': sample,
                'src': src_prepare(sample[0]),
                'target': sample[1]
            }
            samples.append(value)

    if shrink_samples > 0:
        samples = random.sample(samples, shrink_samples)

    df = pd.DataFrame.from_dict(samples)
    # print(df.iloc[0])

    return df



def get_dataset(data, train_prcnt=0.8, val_prcnt=0.1, test_prcnt=0.1):
    assert train_prcnt + val_prcnt + test_prcnt == 1

    n = len(data)
    train_idx = int(n*train_prcnt)
    val_idx = train_idx + int(n*val_prcnt)

    tr_data, v_data, ts_data = data.iloc[0:train_idx], data.iloc[train_idx:val_idx], data.iloc[val_idx:]
    return tr_data, v_data, ts_data



