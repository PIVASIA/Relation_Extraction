import logging
import os

import numpy as np
import torch
from relex.datautils import load_id2label
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, RobertaConfig, get_linear_schedule_with_warmup

from phobert_em.model import RobertaConcatAll, RobertaEntityStarts

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, tokenizer, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.output_dir = args.save_output
        self.id2label = load_id2label(args.id2label)
        self.num_labels = len(self.id2label)
        
        self.config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task="VLSP2020-Relex",
            id2label={str(i): label for i, label in self.id2label.items()},
            label2id={label: i for i, label in self.id2label.items()},
        )
        if self.args.model_type == "es":
            self.model = RobertaEntityStarts.from_pretrained(args.model_name_or_path, config=self.config)
        elif self.args.model_type == "all":
            self.model = RobertaConcatAll.from_pretrained(args.model_name_or_path, config=self.config)
        
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def evaluate(self):
        labels = [lb for lb in sorted(self.id2label.keys()) if self.id2label[lb] != 'OTHER']
        
        dataset = self.test_dataset

        eval_sampler = SequentialSampler(dataset)
        # print(list(eval_sampler))
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        
        # Eval!
        logger.info("***** Running evaluation on validation dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        
        self.model.eval()
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_ids": batch[4],
                    "e2_ids": batch[5],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        # # print('pred',preds)
        # eval_loss = eval_loss / nb_eval_steps
        # results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)

        true_labels = [self.id2label[i] for i in out_label_ids]
        predictions = [self.id2label[i] for i in preds]
        text_labels = [self.id2label[lb] for lb in labels]
        # print("**** Classification Report ****")
        # print(len(predictions))
        with open(self.output_dir, 'a+', encoding='utf-8') as opt:
            for i in predictions:
                opt.write(i+'\n')
        return predictions

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        
        self.args = torch.load(os.path.join(self.args.model_dir, "training_args.bin"))
        if self.args.model_type == "es":
            self.model = RobertaEntityStarts.from_pretrained(self.args.model_dir, config=self.config)
        elif self.args.model_type == "all":
            self.model = RobertaConcatAll.from_pretrained(self.args.model_dir, config=self.config)
            
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
