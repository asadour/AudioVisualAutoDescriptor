import json
import openai

openai.api_key = "sk-ezabQjSIJwqVUKgawXBLT3BlbkFJj8Ck1QCjhzdT5aCyOUrK"


class chatGPTSummary:
    def __init__(self, text):
        self.prompt = "Summarize following in max 10 words: \n" + text
        response = str(self.getSummary())

        y = json.loads(response)
        self.result = y["text"].replace("\n", "")

    def getSummary(self, model_engine="text-davinci-003"):
        temperature = 0.9
        tokens = 3100
        try:
            response = openai.Completion.create(engine=model_engine, prompt=self.prompt, max_tokens=tokens, n=1,
                                                stop=None,
                                                temperature=temperature)
        except:
            print("OpenAI key is not valid!")
            exit(0)

        else:
            print("OpenAI key is valid!")

        return response["choices"][0]
