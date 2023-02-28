import pathlib
import sys

directory = pathlib.Path(__file__).absolute()
sys.path.append(str(directory.parent.parent))
import utils

from transformers import T5ForConditionalGeneration
import t5trainer

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute())
# MODEL_PATH = BASE_PATH + "/models/t5-checkpoints/best-checkpoint.ckpt"
# MODEL_PATH = BASE_PATH + "/models/t5-large-basic_synth_15000_10-checkpoints/best-checkpoint.ckpt"
MODEL_PATH = BASE_PATH + "/models/t5-large-basic_synth_30000_12-epoch3.ckpt"
# MODEL_PATH = BASE_PATH + "/models/t5-base-basic_synth_15000_10-checkpoints/best-checkpoint.ckpt"
# MODEL_PATH = BASE_PATH + "/names-model.ckpt"


trained_model = t5trainer.TransModel.load_from_checkpoint(MODEL_PATH)
trained_model.freeze()


tokenizer = t5trainer.get_tokenizer()

sentences = [
    [('Barack Obama', 'b. obama'), ('Davood Rafiei', 'd. rafiei'), 'Arash Dargahi'],
    [('Barack Hossein Obama', 'b. h. obama'), ('Davood rafiei', 'd. rafiei'), 'Arash Dargahi Nobari'],
    # [('dargahi', 'ihagrad'), ('rafiei', 'ieifar'), 'kamalloo'],
    [('587-165-1245', '5871651245'), ('143-564-3487', '1435643487'), '874-098-4587'],
    [('Dried Apple', 'Apple'), ('Nice Cherry', 'Cherry'), 'Best Peach'],
]

values = []
for sentence in sentences:
    values.append(utils.src_prepare(sentence))



def transform(value, model):
    inps = t5trainer.TransformationDataset.get_src_encoding(value, tokenizer)
    gen_ids = model.generate(
        input_ids=inps['input_ids'],
        attention_mask=inps['attention_mask'],
        num_beams=1,
        max_length=50,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in gen_ids
    ]
    return "".join(preds)


for value in values:
    print(transform(value, trained_model.model))


print("-------------")
model = T5ForConditionalGeneration.from_pretrained(t5trainer.MODEL_NAME, return_dict=True)

for value in values:
    print(transform(value, model))

