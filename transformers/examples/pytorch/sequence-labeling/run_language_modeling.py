# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm import tqdm
import csv
from copy import deepcopy
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DefaultDataCollator,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
)
from datasets import load_metric
import segeval as seg

metric1 = load_metric("metric/accuracy.py")
metric2 = load_metric("metric/precision.py")
metric3 = load_metric("metric/recall.py")
metric4 = load_metric("metric/f1.py")
def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def pk(h,gold):
    h_new=''
    gold_new=''
    for n,x in enumerate(h):
        h_new+=str(int(x))
    for n,x in enumerate(gold):
        gold_new+=str(int(x))
    h_new = seg.convert_nltk_to_masses(h_new)
    gold_new = seg.convert_nltk_to_masses(gold_new)
    pk = seg.pk(h_new, gold_new)
    return pk
def write2excel(list_,name):
    with open("./results/"+name+'.csv', 'w') as output:
      writer = csv.writer(output)
      writer.writerow([k for k,v in list_[0].items()])
      for dict_ in list_:
              writer.writerow([v for k,v in dict_.items()])

import  numpy as np
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_concept_file: Optional[str] = field(
        default='train_concept.txt', metadata={"help": "The concept training data file (a text file)."}
    )
    train_index_file: Optional[str] = field(
        default='train_index.txt', metadata={"help": "The concept training index file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    eval_concept_file: Optional[str] = field(
        default='dev_concept.txt', metadata={"help": "The concept eavl data file (a text file)."}
    )
    eval_index_file: Optional[str] = field(
        default='dev_index.txt', metadata={"help": "The concept dev index file (a text file)."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    output_information: bool = field(
        default=False,
        metadata={"help": "Whether output attention figures."},
    )

    con_loss: bool = field(
        default=False,
        metadata={"help": "Whether con learning."},
    )

    add_entity_type: bool = field(
        default=False,
        metadata={"help": "Whether use entity type."},
    )

    eval_label2: bool = field(
        default=False,
        metadata={"help": "Whether eval_label2."},
    )

    yuzhi: float = field(
        default=0.5, metadata={"help": "threshold."}
    )

    out_file_name: Optional[str] = field(
        default='1', metadata={"help": "The output of results."}
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    choice: int = field(
        default=0, metadata={"help": "mask_size for loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )



class LMDataset(torch.utils.data.Dataset):
    def __init__(self, block_size, data_path, tokenizer: PreTrainedTokenizer, con_loss, entity_type,eval_label2):

        self.block_size=block_size
        self.data_path=data_path
        self.con_loss=con_loss
        self.add_entity_type=entity_type
        self.eval_label2=eval_label2
        self.text_path = data_path+'text/'
        self.label1_path=data_path+'label1/'
        self.label2_path = data_path + 'label2/'
        self.entity_type_path = data_path + 'name_entity_file/'
        self.tokenizer=tokenizer
        self.textfiles = self._get_files()
    def _get_files(self):
        files=[]
        for filename in os.listdir(self.text_path):
            files.append(filename)
        return files
    def clean_paragraph(self, paragraph):
        cleaned_paragraph = paragraph.strip('\n')
        return cleaned_paragraph

    def _get_exmples(self, path1, tokenizer, block_size):

        #logger.info("Creating features from dataset file at %s", path1)

        with open(self.text_path+path1,'r',encoding='utf-8') as f:
            sentence_list = f.readlines()
        if self.eval_label2:
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]
        else:
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]

        token_list = [torch.tensor(x, dtype=torch.long) for x in
                      tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, max_length=block_size)[
                          "input_ids"]]
        type_list=None

        return token_list, label1, label2, type_list

    def __len__(self):
        return len(self.textfiles)

    def __getitem__(self, i):
        path = self.textfiles[i]

        token_list,label1,label2, type_list=self._get_exmples(path, self.tokenizer, self.block_size)
        if self.con_loss:
            if self.add_entity_type:
                return [token_list, torch.tensor(label1, dtype=torch.long), torch.tensor(label2, dtype=torch.long), type_list, None,None]
            else:
                return [token_list,torch.tensor(label1, dtype=torch.long),torch.tensor(label2, dtype=torch.long),None, None,None]
        elif self.add_entity_type:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, type_list, None,None]
        else:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, None, None,None]
class LM_train_Dataset(torch.utils.data.Dataset):
    def __init__(self, block_size, data_path, tokenizer: PreTrainedTokenizer, con_loss, entity_type,eval_label2,choice):

        self.block_size=block_size
        self.data_path=data_path
        self.con_loss=con_loss
        self.add_entity_type=entity_type
        self.eval_label2=eval_label2
        self.text_path = data_path+'text/'
        self.label1_path=data_path+'label1/'
        self.label2_path = data_path + 'label2/'
        self.entity_type_path = data_path + 'name_entity_file/'
        self.tokenizer=tokenizer
        self.textfiles = self._get_files()
        self.choose_prob=None
        self.mask_prob=[0,0.25,0.5,0.75,1]
        self.epoch=0
        self.choice=choice
        if self.choice == 0:
            self.epochs=[2,6,10]
        elif self.choice == 1:
            self.epochs = [5, 15, 40]
        else:
            pass
    def get_seg(self,label):
        if self.choice<=1:
            if self.epoch <= self.epochs[0]:
                self.choose_prob = [0, 1, 0, 0, 0]
            elif self.epoch <= self.epochs[1]:
                self.choose_prob = [0, 0, 1, 0, 0]
            elif self.epoch <= self.epochs[2]:
                self.choose_prob = [0, 0, 0, 1, 0]
            else:
                self.choose_prob = [0, 0, 0, 0, 1]
        else:
            if self.choice==2:
                self.choose_prob = [0, 0.1, 0.3, 0.6, 0]
            elif self.choice==3:
                self.choose_prob = [0, 0.3, 0.3, 0.4, 0]
            elif self.choice==4:
                self.choose_prob = [0, 0.25, 0.25, 0.25, 0.25]
            elif self.choice==5:
                self.choose_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
        p = np.array(self.choose_prob)
        index = np.random.choice([0, 1, 2, 3, 4], p=p.ravel())
        mask_prob=self.mask_prob[index]
        location=torch.nonzero(label)
        mask_num=int(mask_prob*len(location))
        if mask_num>0:
            mask_index=np.random.choice(location.view(-1).numpy(),mask_num,replace=False)
            label[mask_index]=0
        return  label
    def _get_files(self):
        files=[]
        for filename in os.listdir(self.text_path):
            files.append(filename)
        return files
    def clean_paragraph(self, paragraph):
        cleaned_paragraph = paragraph.strip('\n')
        return cleaned_paragraph

    def _get_exmples(self, path1, tokenizer, block_size):

        #logger.info("Creating features from dataset file at %s", path1)

        with open(self.text_path+path1,'r',encoding='utf-8') as f:
            sentence_list = f.readlines()
        if self.eval_label2:
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]
        else:
            with open(self.label1_path + path1, 'r', encoding='utf-8') as f:
                label1 = [int(x) for x in f.readlines()]
            with open(self.label2_path + path1, 'r', encoding='utf-8') as f:
                label2 = [int(x) for x in f.readlines()]

        token_list = [torch.tensor(x, dtype=torch.long) for x in
                      tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, max_length=block_size)[
                          "input_ids"]]
        type_list=None

        return token_list, label1, label2, type_list

    def __len__(self):
        return len(self.textfiles)

    def __getitem__(self, i):
        path = self.textfiles[i]

        token_list,label1,label2, type_list=self._get_exmples(path, self.tokenizer, self.block_size)
        seg_now = self.get_seg(torch.tensor(label1, dtype=torch.long))
        if self.con_loss:
            if self.add_entity_type:
                return [token_list, torch.tensor(label1, dtype=torch.long), torch.tensor(label2, dtype=torch.long), type_list, None, seg_now]
            else:
                return [token_list,torch.tensor(label1, dtype=torch.long),torch.tensor(label2, dtype=torch.long),None, None, seg_now]
        elif self.add_entity_type:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, type_list, None, seg_now]
        else:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, None, None, seg_now]

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LMDataset(
            tokenizer=tokenizer,  data_path=file_path, block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2
        )

def get_train_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LM_train_Dataset(
            tokenizer=tokenizer,  data_path=file_path, block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2, choice=args.choice
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, con_loss=args.con_loss, entity_type=args.add_entity_type, eval_label2=args.eval_label2, choice=args.choice
        )

yuzhi=0.5
# Following the previous work (Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence),
# we implement the same pk metric based on the open source code here (https://github.com/kelvinlo-uni/Transformer-squared).
def compute_metrics(eval_pred,all_length):
    global yuzhi
    all_start=[0]
    for i in range(len(all_length)):
        all_start.append(all_start[i]+all_length[i])
    logits, labels = eval_pred

    logits = logits.reshape(-1, 2)
    labels = labels.reshape(-1)
    lines = []
    i = 0
    for line in range(labels.shape[0]):
        if labels[line] != -100:
            lines.append(line)
    logits = logits[lines]
    labels = labels[lines]
    predictions = np.argmax(logits, axis=-1)
    for i in range(len(predictions)):
        if logits[i][1] > yuzhi:
            predictions[i] = 1
    dic1 = metric1.compute(predictions=predictions, references=labels)
    dic2 = metric2.compute(predictions=predictions, references=labels)
    dic3 = metric3.compute(predictions=predictions, references=labels)
    dic4 = metric4.compute(predictions=predictions, references=labels)
    pk_mean = 0
    for i in range(len(all_length)):
        start = all_start[i]
        end = all_start[i + 1]
        h=[int(x) for x in predictions[start:end]]
        t=labels[start:end]
        pk_mean += float(pk(h, t)) / len(all_length)
    dic5 = {'pk': pk_mean}
    dic1.update(dic2)
    dic1.update(dic3)
    dic1.update(dic4)
    dic1.update(dic5)
    return dic1

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.out_file_name1=data_args.out_file_name
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    config.label_num=training_args.label_num
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    train_dataset = (
        get_train_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator =DataCollatorWithPadding(tokenizer)

    global yuzhi
    yuzhi=data_args.yuzhi

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    torch.autograd.set_detect_anomaly(True)
    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        tokenizer.save_pretrained(training_args.output_dir)
        trainer.train(model_path=model_path)


if __name__ == "__main__":
    main()