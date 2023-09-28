import json
import openai

openai.api_key = "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"

class chatGPTSummary:
    def __init__(self, text):
        self.prompt = "Summarize following in max 10 words: \n"+text
        response = str(self.getSummary())

        y = json.loads(response)
        self.result = y["text"].replace("\n", "")

    def getSummary(self, model_engine="text-davinci-003"):
        temperature = 0.9
        tokens = 3100
        response = openai.Completion.create(engine=model_engine, prompt=self.prompt, max_tokens=tokens, n=1, stop=None,
                                            temperature=temperature)
        return response["choices"][0]



