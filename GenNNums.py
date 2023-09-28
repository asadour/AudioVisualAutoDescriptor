class GenN:
    def __init__(self, times, maxNum):
        self.times = times
        self.Max = maxNum
        self.Numbers = []
        self.generateNRandomNums()

    def generateNRandomNums(self):
        total = self.Max
        n = self.times
        import random
        counter = 0
        per_n = int(float(total) / float(n))
        numbers = []
        for i in range(n):
            numbers.append(random.randint(counter + 1, counter + per_n))
            counter += per_n
        self.Numbers = numbers
        return numbers


# The following is for testing reasons only !
from SummarizerNLTK import BestSummarizer as bs
v = bs("Person in the bathroom mirror. The man in the mirror.").Paraphrased
#print(v)
