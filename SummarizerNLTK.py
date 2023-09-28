from transformers import *
import logging
logging.basicConfig(level='ERROR')
logging.disable(level='WARNING')
import warnings
warnings.filterwarnings("ignore")
import shutup
shutup.please()

class BestSummarizer:
    def __init__(self, text):
        self.sentence = str(text)
        self.Paraphrased = None # receive paraphrased text
        self.model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
        self.tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
        self.get_paraphrased_sentences()

    def get_paraphrased_sentences(self, num_return_sequences=1, num_beams=1):
        # tokenize the text to be form of a list of token IDs
        inputs = self.tokenizer([self.sentence], truncation=True, padding="longest", return_tensors="pt")
        # generate the paraphrased sentences
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        # decode the generated sentences using the tokenizer to get them back to text
        paraphrased = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.Paraphrased = paraphrased[0]
        return paraphrased
