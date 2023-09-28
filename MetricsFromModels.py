from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd


class getRougeFromModels:
    def __init__(self, dataset, rows,separator, modelp="facebook/bart-large", decoder_type="bart"):

        trainedModel = modelp

        model_args = Seq2SeqArgs()
        model_args.num_train_epochs = 1
        model_args.no_save = False
        model_args.evaluate_generated_text = True
        model_args.evaluate_during_training = True
        model_args.evaluate_during_training_verbose = False
        model_args.overwrite_output_dir = True
        model1 = Seq2SeqModel(
            encoder_decoder_type=decoder_type,
            encoder_decoder_name=trainedModel,
            args=model_args,
            use_cuda=False
        )
        if rows < 0:
            Ds = pd.read_csv(dataset, header=0, sep=separator)
        else:
            Ds = pd.read_csv(dataset, header=0, sep=separator)[:rows]
        Ds = Ds.sample(frac=1)
        readDs = Ds["input"]

        readDsTarget = Ds["target"]
        print(readDsTarget)
        counter = 0
        metrics = 0.0
        import evaluate
        rouge = evaluate.load('rouge')


        for row in readDs:
            print(row)
            pred = model1.predict([row])
            print(pred)

            results = rouge.compute(predictions=[pred[0]], references=[readDsTarget.values[counter]], tokenizer=lambda x: x.split())
            print(results)
            jsonstr = str(results).replace("{","").replace("}","").replace(" ","").split(",")[0].split(":")[1]

            counter += 1
            print(counter)
            metrics += float(jsonstr)

        self.result = metrics/counter
        print(metrics/counter)

class getRougeFromModelPegasus:
    def __init__(self, dataset, rows, separator):
        self.dataset = dataset
        self.runPegasus(rows, separator)
    def runPegasus(self, rows, separator):

        counter = 0
        metrics = 0.0
        import evaluate
        rouge = evaluate.load('rouge')
        if rows < 0:
            Ds = pd.read_csv(self.dataset, header=0, sep=separator)
        else:
            Ds = pd.read_csv(self.dataset, header=0, sep=separator)[:rows]
        Ds = Ds.sample(frac=1)
        readDs = Ds["input"]

        readDsTarget = Ds["target"]
        for row in readDs:
            print(row)
            pred = self.getSummary(row)
            print(pred)

            results = rouge.compute(predictions=[pred], references=[readDsTarget.values[counter]],
                                    tokenizer=lambda x: x.split())
            print(results)
            jsonstr = str(results).replace("{", "").replace("}", "").replace(" ", "").split(",")[0].split(":")[1]

            counter += 1
            print(counter)
            metrics += float(jsonstr)

        self.result = metrics / counter
        print(metrics / counter)

    def getSummary(self, text):
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        import torch

        src_text = [ text ]

        model_name = "sshleifer/distill-pegasus-xsum-16-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
        batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text[0]