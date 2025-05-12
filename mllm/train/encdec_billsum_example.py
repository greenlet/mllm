import os
from pathlib import Path
import sys
from typing import Optional, cast
if '..' not in sys.path: sys.path.append('..')

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


def train_billsum_encdec():
    billsum = load_dataset("billsum")
    checkpoint = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prefix = "summarize: "
    # max_inputs_len, max_summary_len = 1024, 128
    max_inputs_len, max_summary_len = 128, 16

    def preprocess_function(example):
        inputs = [prefix + doc for doc in example["text"]]
        model_inputs = tokenizer(inputs, max_length=max_inputs_len, truncation=True)

        labels = tokenizer(text_target=example["summary"], max_length=max_summary_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    out_dpath = Path(os.path.expandvars('$HOME')) / 'data/billsum_finetune'
    print(out_dpath)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dpath),
        # eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        # bf16=True,
        # fp16=False, #change to bf16=True for XPU
        push_to_hub=False,
    )

    tokenized_billsum = billsum.map(preprocess_function, batched=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        # processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    train_billsum_encdec()

