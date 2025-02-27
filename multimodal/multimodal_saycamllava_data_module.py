from pathlib import Path
from typing import Any, Tuple
from collections import Counter
import os
import glob
import itertools
import json
import csv
import random
import re
import shutil
import time
import cv2 as cv

#unused? 
#import imageio
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import os.path
import sys

#sys.path.append('/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/multimodal')

from multimodal.multimodal_data_module import MultiModalDataset, \
    MultiModalDataModule, read_vocab, load_data, load_and_print_info, \
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, \
    IMAGE_H, IMAGE_W, multiModalDataset_collate_fn

from multimodal.utils import *

import spacy
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directories and filename - need to make sure these are consistent elsewhere and 
#set up symbolic link relative to working directory 
TRAIN_DATA_DIR = Path("/projectnb/ivc-ml/wsashawn/LLaVA/llava_SAYCam_mixed_data.json")
VAL_DATA_DIR = Path("/projectnb/ivc-ml/wsashawn/SAYCam/val.csv")
TEST_DATA_DIR = Path("/projectnb/ivc-ml/wsashawn/SAYCam/test.csv")
VOCAB_FILENAME = Path("/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/arjun_misc/vocab.json")
SAYCAM_ROOT = "/projectnb/ivc-ml/wsashawn/SAYCam/train_5fps"
LLAVA_ROOT = "/projectnb/ivc-ml/wsashawn/dataset/llava_pretrain_558k/images"


class MultiModalSAYCamLLaVADataset(MultiModalDataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset
    and data from LLaVA pretraining modified for baby training. 
    """

    def __init__(self, data_path, vocab, transform):
        """
        - Training data is .json file
        - Val/Test data is .csv file 
        """
        super().__init__()
        self.data = self._load_data(data_path)
        #print(self.data)
        self.vocab = vocab
        self.transform = transform

        #load tokenizer for LLaVA
        self.nlp = spacy.load(
        'en_core_web_sm',
            exclude=[
                'attribute_ruler', 'lemmatizer', 'ner',
                'senter', 'parser', 'tagger', 'tok2vec']
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    #NEED TO IMPLEMENT THIS - done
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        """

        # get utterance and image 
        utterance = self.data[idx]["caption"]
        img_path = self.data[idx]["image"]

        # looks like tokenization is just white space again but need to make sure this 
        # is consistent with our SayCam and LLava tokenization 
        if img_path.startswith('llava'):
            utterance_words = [token.text for token in self.nlp.tokenizer(utterance)]
        else:
            utterance_words = utterance.split()
        
        utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)

        #ensure lower case when going from token to id
        utterance_idxs = torch.tensor([self.vocab.get(
            word.lower(), UNK_TOKEN_ID) for word in utterance_words], dtype=torch.long)

        #train images need this replacement, val and test don't
        if img_path.startswith('llava'):
            img_filename = Path(img_path.replace("llava_pretrain", LLAVA_ROOT))
        elif img_path.startswith('SAYCam'):
            img_filename = Path(img_path.replace("SAYCam", SAYCAM_ROOT))
        else:
            img_filename = Path(img_path)

        img = Image.open(img_filename).convert("RGB")
        #LLaVA images need to be resized
        img = img.resize((IMAGE_W, IMAGE_H))

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length, [utterance]

    #NEED TO IMPLEMENT THIS - done
    def _load_data(self, data_path):
        """
        Format data based on .json (train) or .csv (val.test)
        """
        if data_path.suffix == '.json':
            return self._process_json(data_path)
        elif data_path.suffix == '.csv':
            return self._process_csv(data_path)
        else:
            raise NotImplementedError

    def _process_json(self, data_path):
        """
        - Process json dataset and format it for training
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Transform to training format - convert to lowercase to match tokenization step
        formatted_data = {
            int(i): {
                "image": entry["image"],
                "caption": entry["conversations"][1]["value"].lower()
            }
            for i, entry in enumerate(data)
        }

        return formatted_data

    def _process_csv(self, data_path):
        """
        - Process csv file containg val and test data
        """
        formatted_data = {}
        
        with open(data_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                formatted_data[index] = {
                    "image": row["image"],
                    "caption": row["text"]
                }
        
        return formatted_data
        

class MultiModalSAYCamLLaVADataModule(MultiModalDataModule):
    """
    A data module created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances. Also includes 
    LLaVA modified pretraining data. 
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        #can save any args if needed here

    #no additional arguments for now
    @staticmethod 
    def add_additional_to_argparse(parser):
        return None

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalSAYCamLLaVADataModule,
                       MultiModalSAYCamLLaVADataModule).add_to_argparse(parser)
        parser = MultiModalSAYCamLLaVADataModule.add_additional_to_argparse(parser)
        return parser

    #Nothing to do here 
    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)

    #NEED TO IMPLEMENT THIS - done
    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    #Overriding this since we don't have eval datasets or the path names used in the parent class
    def setup(self, *args, **kwargs) -> None:
        print("Calling setup for Saycam + LLavA dataset!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val/test) -> they use multiple val and test methods (loss and accuracy)
        # self.eval_datasets = self.create_eval_datasets(vocab)

    #NEED TO IMPLEMENT THIS
    #Create train, val, test datasets. Dataloading and collate is handled in parent class
    def create_datasets(self, vocab):
        datasets = {}
        print("Creating datasets for Saycam + LLava mixed data...")
        
        stage_splits = [("train", TRAIN_DATA_DIR, self.transform),
                        ("val", VAL_DATA_DIR, self.base_transform),
                        ("test", TEST_DATA_DIR, self.base_transform)]

        for split, data_path, transform in stage_splits:
            dataset = MultiModalSAYCamLLaVADataset(
                data_path,
                vocab,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets

    
    #Need to override dataloader creation since we don't do Labeled-S eval during training (self.eval_datasets)
    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        val_dataloader = DataLoader(
            self.datasets['val'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return val_dataloader

    #Need to override dataloader creation since we don't do Labeled-S eval during training  (self.eval_datasets)
    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        test_dataloader = DataLoader(
            self.datasets['test'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return test_dataloader


#This just runs prepare_data from parent class - not needed 
# if __name__ == "__main__":
#     load_and_print_info(MultiModalSAYCamDataModule)

# Some test code
# data_module = MultiModalSAYCamLLaVADataModule()
# data_module.setup()
# print(data_module.datasets["test"][1532])