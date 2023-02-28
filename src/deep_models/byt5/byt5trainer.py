import pathlib
import torch.optim
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, AutoTokenizer, AutoModelForSeq2SeqLM, \
    ByT5Tokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import sys
directory = pathlib.Path(__file__).absolute()
sys.path.append(str(directory.parent.parent))
import utils

MAX_SRC_LEN = 500
MAX_TARGET_LEN = 60


BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute())


# DATASET_NAME = "ffaj_10rt_samples"
DATASET_NAME = "basic_synth_40000_10"
# MODEL_NAME = "t5-large"
MODEL_NAME = "google/byt5-base"
BATCH_SIZE = 5
MAX_EPOCHS = 10

SAMPLES_PATH = BASE_PATH + f"/data/SampleSets/{DATASET_NAME}.json"
CHECKPOINTS_PATH = BASE_PATH + f"/models/{MODEL_NAME.replace('google/', '')}-{DATASET_NAME}-checkpoints"
CHECKPOINTS_FILENAME = "best-checkpoint"

TR_TOKEN = utils.TR_TOKEN
EOE_TOKEN = utils.EOE_TOKEN
ESP_TOKENS = [TR_TOKEN, EOE_TOKEN]

SHRINK_SAMPLES = -1

INIT_DICT_SIZE = None
NUM_DL_WORKER = 4


class TransformationDataset(Dataset):
    def __init__(self, data, tokenizer, src_max_token_len=MAX_SRC_LEN, target_max_token_len=MAX_TARGET_LEN):
        self.tokenizer = tokenizer
        self.data = data
        self.src_max_token_len = src_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)


    @staticmethod
    def get_src_encoding(src, tknizer):
        tknizer.truncation_side = "left"
        return tknizer(
            src,
            # samples[0]['more_info_on_inputs'],
            max_length=MAX_SRC_LEN,
            padding="max_length",
            # truncation="only_second",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

    @staticmethod
    def get_labels(target, tknizer):
        tknizer.truncation_side = "right"
        target_encoding = tknizer(
            target,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        # print(tokenizer.decode((target_encoding["input_ids"]).squeeze()))
        labels = target_encoding["input_ids"]
        # labels[labels == 0] = -100

        return labels


    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        src_encoding = self.get_src_encoding(data_row['src'], self.tokenizer)
        # print(tokenizer.decode((src_encoding["input_ids"]).squeeze()))

        labels = self.get_labels(data_row['target'], self.tokenizer)
        # print(labels)


        res = {
            'sample': data_row['sample'],
            'src': data_row['src'],
            'target': data_row['target'],
            'input_ids': src_encoding['input_ids'].flatten(),
            'attention_mask': src_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }
        return res



class TransDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, src_max_token_len=MAX_SRC_LEN, target_max_token_len=MAX_TARGET_LEN ):
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.src_max_token_len = src_max_token_len
        self.target_max_token_len = target_max_token_len



    def setup(self, stage=None):
        self.train_dataset = TransformationDataset(train_df, tokenizer, self.src_max_token_len, self.target_max_token_len)
        self.val_dataset = TransformationDataset(val_df, tokenizer, self.src_max_token_len, self.target_max_token_len)
        self.test_dataset = TransformationDataset(test_df, tokenizer, self.src_max_token_len, self.target_max_token_len)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_DL_WORKER)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_DL_WORKER)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_DL_WORKER)



class TransModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
        # return AdamW(self.parameters(), lr=0.0001)




def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = ByT5Tokenizer(extra_ids=5)
    INIT_DICT_SIZE = len(tokenizer)
    # special_tokens_dict = {'additional_special_tokens': ESP_TOKENS + tokenizer.all_special_tokens}
    special_tokens_dict = {'additional_special_tokens': ESP_TOKENS}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer



if __name__ == "__main__":

    df = utils.get_dataframe(SAMPLES_PATH, SHRINK_SAMPLES)
    tokenizer = get_tokenizer()

    train_df, val_df, test_df = utils.get_dataset(df)


    # inp = df.iloc[0]['src']
    # inp = f"ab{utils.TR_TOKEN}cdD4@{utils.EOE_TOKEN}"
    # test = tokenizer(inp)
    # print(inp)
    # print(test)
    # words = [tokenizer.decode(inp_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    #          for inp_id in test["input_ids"]]
    # print(words)
    #
    # exit()


    data_module = TransDataModule(train_df, val_df, test_df, tokenizer, MAX_SRC_LEN, MAX_TARGET_LEN)
    data_module.setup()


    # model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    # dl = data_module.val_dataloader()
    # for batch_idx, batch in enumerate(dl):
    #     print(batch)
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['labels']
    #     loss, output = model(input_ids, attention_mask, labels)
    #     print(output)
    #     exit()


    model = TransModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_PATH,
        filename=CHECKPOINTS_FILENAME,
        save_top_k=1,
        verbose=True,
        monitor="val loss",
        mode="min"
    )


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, LitProgressBar()],
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        devices=1
    )


    trainer.fit(model, data_module)



