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

import segeval as seg
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


import  numpy as np
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
English=False
yuzhi=0.5
dataset_size=-1

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
        default='dev_concept.txt', metadata={"help": "The concept eval data file (a text file)."}
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
        metadata={"help": "Whether use topic labels."},
    )
    add_entity_type: bool = field(
        default=False,
        metadata={"help": "Whether use entity type."},
    )

    eval_label2: bool = field(
        default=False,
        metadata={"help": "Whether eval_label2."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )

    English: bool = field(
        default=False,
        metadata={
            "help": "The language of the tested dataset."
        },
    )

    dataset_size: int = field(
        default=-1,
        metadata={
            "help": "Test dataset size."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, block_size, data_path, tokenizer: PreTrainedTokenizer, con_loss, entity_type, eval_label2):

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

        token_list = [torch.tensor(x, dtype=torch.long) for x in tokenizer.batch_encode_plus(sentence_list, add_special_tokens=True, max_length=block_size)["input_ids"]]

        position_list=[]

        for sentence_tokens in token_list:
            sentence_position_list=[-1,0]
            for token in sentence_tokens:
                if token.item() not in [101,102]:

                    s=tokenizer.convert_ids_to_tokens(token.unsqueeze(0))[0]
                    s=s.replace('##','')
                    length=len(s)

                    if s.find('[')!=-1 and s.find(']')!=-1:
                        length=1
                    sentence_position_list.append(sentence_position_list[-1]+length)

            sentence_position_list[-1]=-1

            position_list.append(sentence_position_list)

        type_list=[]
        return token_list, label1, label2, type_list

    def __len__(self):
        return len(self.textfiles)

    def __getitem__(self, i):
        path = self.textfiles[i]

        token_list,label1,label2, type_list=self._get_exmples(path, self.tokenizer, self.block_size)
        if self.con_loss:
            if self.add_entity_type:
                return [token_list, torch.tensor(label1, dtype=torch.long), torch.tensor(label2, dtype=torch.long), type_list, path,None]
            else:
                return [token_list,torch.tensor(label1, dtype=torch.long),torch.tensor(label2, dtype=torch.long),None, path,None]
        elif self.add_entity_type:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, type_list, path,None]
        else:
            return [token_list, torch.tensor(label1, dtype=torch.long), None, None, path,None]

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
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            try:
                os.remove(path_file)
            except:
                pass
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    try:
                        os.remove(path_file2)
                    except:
                        pass

def pk(h,gold):
    global English
    h_new=''
    gold_new=''
    for n,x in enumerate(h):
        h_new+=str(int(x))
    for n,x in enumerate(gold):
        gold_new+=str(int(x))
    h_new = seg.convert_nltk_to_masses(h_new)
    gold_new = seg.convert_nltk_to_masses(gold_new)
    dic={}
    # SegFormer uses window_size=10 on the German datasets, consistent with Transformer^2, 
    # otherwise the results on the German datasets cannot exceed the results reported by Transformer^2.
    # All experiments except for German experiments use the default window size, which is the standard setting.
    if English:
        pk = seg.pk(h_new, gold_new)
    else:
        pk = seg.pk(h_new, gold_new, window_size=10)
    return pk

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
    #Remove the padding value.
    for line in range(labels.shape[0]):
        if labels[line] != -100:
            lines.append(line)
    logits = logits[lines]
    labels = labels[lines]
    predictions = np.argmax(logits, axis=-1)
    for i in range(len(predictions)):
        if logits[i][1] > yuzhi:
            predictions[i] = 1
    pk_mean = 0
    # Following the previous work (Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence),
    # We calculate the pk value and average it for each article.
    for i in range(dataset_size):
        start = all_start[i]
        end = all_start[i + 1]
        h=[int(x) for x in predictions[start:end]]
        t=labels[start:end]
        pk_mean += float(pk(h, t)) / dataset_size
    dic1 = {'pk': pk_mean}
    return dic1
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global English
    English=data_args.English
    global dataset_size
    dataset_size=data_args.dataset_size

    # Load pretrained model and tokenizer

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
    config.label_num = training_args.label_num
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

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

    # Get datasets
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
        if training_args.do_eval
        else None
    )

    data_collator =DataCollatorWithPadding(tokenizer)

    #
    # # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluation
    logger.info("*** Evaluate ***")
    eval_output = trainer.evaluate(output_figures=data_args.output_information)
    print(eval_output)


if __name__ == "__main__":
    main()