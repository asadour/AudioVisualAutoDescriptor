import os

import evaluate
import nltk
import numpy as np
import torch

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoint = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = model.to(device)

max_input_length = 4096
max_target_length = 4096


class FineTuneBART:
    def __init__(self, folderName, datasetlist):
        dataset = load_dataset("csv", data_files={"train": datasetlist[0], "validation": datasetlist[1],
                                                  "test": datasetlist[2]}, sep=",", keep_default_na=False)
        dataset = dataset.remove_columns(['0', '1', '2', '3'])
        print(dataset)

        dataset['train'] = dataset['train']  # .shuffle(seed=42).select(range(10000))
        dataset['validation'] = dataset['validation']  # .shuffle(seed=42).select(range(1000))
        dataset['test'] = dataset['test']  # in.shuffle(seed=42).select(range(2000))
        self.show_samples(dataset)
        os.environ["WANDB_DISABLED"] = "true"

        tokenized_dataset = DatasetDict()

        # Tokenize each split of the dataset
        for split_name in dataset.keys():
            split = dataset[split_name]
            tokenized_split = split.map(self.preprocess_function, batched=True)
            tokenized_dataset[split_name] = tokenized_split

        # Verify the tokenized dataset
        print("Tokenized documents in the train split:", tokenized_dataset['train']['input_ids'][:5])
        print("Tokenized summaries in the train split:", tokenized_dataset['train']['labels'][:5])

        print(tokenized_dataset)


        batch_size = 1 #16
        num_train_epochs = 2

        # Show the training loss with every epoch
        logging_steps = len(tokenized_dataset["train"]) // batch_size
        model_name = model_checkpoint.split("/")[-1]

        args = Seq2SeqTrainingArguments(
            output_dir=f"{model_name}-"+folderName,
            evaluation_strategy="steps",
            learning_rate=2.0e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.1,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            logging_steps=logging_steps,
            push_to_hub=False,
            load_best_model_at_end=True,
            eval_steps=25
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        tokenized_dataset = tokenized_dataset.remove_columns(
            dataset["train"].column_names
        )
        from transformers import EarlyStoppingCallback

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()

        trainer.save_model(args.output_dir)

        predictions = trainer.predict(tokenized_dataset["test"])
        print(predictions)

    def show_samples(self, dataset, num_samples=3, seed=42):
        sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
        for example in sample:
            print(f"\n'>> Summary: {example['target']}'")
            print(f"'>> Document: {example['input']}'")

    # Tokenization function
    def preprocess_function(self, examples):
        model_inputs = tokenizer(examples["input"], max_length=1024, truncation=True, )
        labels = tokenizer(examples["target"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        from rouge_score import rouge_scorer
        rouge_score = evaluate.load("rouge")
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
        scr = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

        for idx in range(len(decoded_labels)):
            results = scr.score(decoded_labels[idx], decoded_preds[idx])
            for k in results:
                print(f'{k} : {results[k]}')

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, round(result[rn] * 100, 2)) for rn in rouge_names)
        print(result)
        print(bleu_results)
        return rouge_dict


