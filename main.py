import os.path
import uuid
import clip
import PIL.Image
import cv2
import pandas as pd
import skimage.io as io
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn

import DatasetConstants
import DatasetConstants as dconst
from DatasetSetup import Dataset
from Settings import *
from SumRow import Row

current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'model_wieghts.pt')

model = None
clip_model = None
preprocess = None
tokenizer = None

globCounter = 0
arrLen = range(0)


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def showImage(image_name):
    img = PIL.Image.open(image_name)
    img.show()


def drawTextToImage(real_image_path, img_name, text, video_name):
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    if not os.path.exists(video_name):
        os.mkdir(video_name)

    if not os.path.exists(video_name + "\\AI"):
        os.mkdir(video_name + "\\AI")

    main_path = video_name + "\\AI"

    img = Image.open(real_image_path)
    I1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype("FreeMono.ttf", 42)

    # Add Text to an image
    I1.text((10, 10), text, (255, 0, 0), font=myFont)
    if showImages:
        img.show()
    img_real_name = img_name.split(".")[0]
    img.save(main_path + "\\" + img_real_name + "_AI.png")


def getImagePath(image_name):
    return test_images_folder + "\\" + image_name


from PIL import Image as img


def sampleMovieClip(videoPath):
    from GenNNums import GenN

    cap = cv2.VideoCapture(videoPath)
    frame_images = []
    counter = 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 120:
        nums = GenN(4, frame_count).Numbers
    else:
        nums = GenN(3, frame_count).Numbers
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        if counter in nums:  # this is the line I added to make it only save one frame
            # every 10
            samplePath = 'Samples\\' + str(uuid.uuid4()) + '.png'
            frame_images.append(samplePath)
            cv2.imread('TestImages\\monkey.png')
            cv2.imwrite(samplePath, frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

    return frame_images


def setMainModels():
    clip_Model, prepProcess = clip.load("ViT-B/32", device=device, jit=False)
    Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    Model = ClipCaptionModel(prefix_length)

    Model.load_state_dict(torch.load("conceptual_weights.pt", map_location=CPU))

    Model = Model.eval()
    Model = Model.to(device)

    return Model, clip_Model, prepProcess, Tokenizer


def processImage(image_name):
    image = io.imread(image_name)
    pil_image = PIL.Image.fromarray(image)
    if showImages:
        showImage(image_name)
    return pil_image


def mainNeuralNet(mod, cl_model, tok, image_name):
    Tokenizer = tok
    Model = mod
    clipModel = cl_model

    pil_image = processImage(image_name)

    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clipModel.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = Model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(Model, Tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(Model, Tokenizer, embed=prefix_embed)
    # sampleVideoIntoFrames("TomJerry.mp4")
    # print('\n')
    # print(generated_text_prefix)
    return generated_text_prefix


def NNforImages(image_name):  # Run neural network for a single image!
    global model, clip_model, preprocess, tokenizer
    model, clip_model, preprocess, tokenizer = setMainModels()
    imagePath = getImagePath(image_name)
    textPrefix = mainNeuralNet(model, clip_model, tokenizer, imagePath)
    print("!!TEST CLIP:!! " + textPrefix)
    return textPrefix


def generateCLIPDescription(imagesPathList, model_, clip_model_, tokenizer_):
    descsList = []

    for imgPath in imagesPathList:
        text_prefix = mainNeuralNet(model_, clip_model_, tokenizer_, imgPath)
        if os.path.exists(imgPath):
            os.remove(imgPath)
        descsList.append(text_prefix)
    return descsList


def remove_dupl(x):
    return list(dict.fromkeys(x))


def processDescrs(descriptions_):
    descriptions = remove_dupl(descriptions_)
    init_text = ""
    for text in descriptions:
        if text.endswith("."):
            init_text += text.capitalize()
        else:
            init_text += text.capitalize() + "."
    return init_text.replace(".", ". ")


def mainProc(dsItemsList, threadId, newDSName):
    model__ = model
    clip_model__ = clip_model
    tokenizer__ = tokenizer

    videoNamesList_ = []
    videoPathsList_ = []
    givenDescrs_ = []
    genDescrs_ = []
    summaryDescrs_ = []
    counter = 1

    for dsItem_ in dsItemsList:
        print("Movie = " + str(counter) + " Thread =  " + threadId)
        videoNamesList_.append(dsItem_.videoName)
        videoPathsList_.append(dsItem_.videoPath)
        givenDescrs_.append(dsItem_.description)
        samplesList_ = sampleMovieClip(dsItem_.videoPath)
        res = generateCLIPDescription(samplesList_, model__, clip_model__, tokenizer__)
        generatedDescr_ = processDescrs(res)
        # summary = BestSummarizer(generatedDescr_).Paraphrased
        genDescrs_.append(generatedDescr_)
        summaryDescrs_.append("")
        counter = counter + 1
        # check_file = os.path.isfile(dsItem.videoPath)
        # print(dsItem.description + " -> " + dsItem.videoPath + " exists?? " + str(check_file))
        # samplesList = sampleMovieClip(dsItem.videoPath)
        # print("Description = ")
        # print(generateCLIPDescription(samplesList, model, clip_model, tokenizer)[0]["summary_text"])
    createExcelWithResults(newDSName + threadId + ".xlsx", givenDescrs_, genDescrs_, videoNamesList_, videoPathsList_,
                           summaryDescrs_)


import xlsxwriter


def createExcelWithResults(excelName, givenDescriptionsList, generatedDescriptionsList, videoNamesList, videoPathsList,
                           paraphrasedDescriptionsList):
    row = 1
    print(givenDescriptionsList)
    workbook = xlsxwriter.Workbook(excelName)
    worksheet = workbook.add_worksheet()
    listLen = range(len(givenDescriptionsList))
    worksheet.write('A' + str(row), '0')
    worksheet.write('B' + str(row), 'target')
    worksheet.write('C' + str(row), 'input')
    worksheet.write('D' + str(row), '1')
    worksheet.write('E' + str(row), '2')
    worksheet.write('F' + str(row), '3')
    row += 1
    for idx in listLen:
        worksheet.write('A' + str(row), videoNamesList[idx])
        worksheet.write('B' + str(row), givenDescriptionsList[idx])
        worksheet.write('C' + str(row), generatedDescriptionsList[idx])
        worksheet.write('D' + str(row), "Summary ->")
        worksheet.write('E' + str(row), paraphrasedDescriptionsList[idx])
        worksheet.write('F' + str(row), videoPathsList[idx])

        row += 1

    workbook.close()
    import csv
    with open(excelName.replace("xlsx", "csv"), 'w', newline='') as file:
        writer = csv.writer(file)

        for idx in listLen:
            writer.writerow(
                [videoNamesList[idx], givenDescriptionsList[idx], generatedDescriptionsList[idx], "Summary ->",
                 paraphrasedDescriptionsList[idx], videoPathsList[idx]])


def prepareDataset(datasetName, numRows, newDSName, perGroup):
    exists = os.path.isdir(DatasetConstants.videosDatasetPath)
    if not exists:
        print(
            "Source folder does not exist! Try to declare a valid source path! \n Consider \"videosDatasetPath\" variable in DatasetConstants python file!")
        exit(0)
    ds = Dataset(datasetName)
    dataSetList = ds.readDataset(numRows)
    for ds in dataSetList:
        print(ds.videoPath)
    # Division of our problem (creation of new dataset) into multiple processes to save time
    n = perGroup
    sublists = [dataSetList[i:i + n] for i in range(0, len(dataSetList), n)]
    arrLen = range(len(sublists))
    bigDS = True
    from multiprocessing import freeze_support
    if not bigDS:
        import multiprocessing
        freeze_support()
        runpool = multiprocessing.Pool()
        for idx in arrLen:
            runpool.apply_async(mainProc(sublists[idx], str(idx + 1), newDSName))
    else:
        import concurrent.futures
        freeze_support()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sublists)) as executor:
            for idx in arrLen:
                executor.submit(mainProc, sublists[idx], str(idx + 1), newDSName)


def CSVToExcel(fileName):
    read_file = pd.read_csv(fileName, header=None)

    newFile = str(fileName).replace("csv", "xlsx")
    read_file.columns = ['0', 'target', 'input', '1', '2', '3']

    read_file.to_excel(newFile,
                       index=False,
                       header=0)
    return newFile


def prepareDatasetStep(dataset, rowsNum, newDSName, perGroup):
    global model, clip_model, preprocess, tokenizer
    model, clip_model, preprocess, tokenizer = setMainModels()
    prepareDataset(dataset, rowsNum, newDSName, perGroup)


def CombineResultsMultiProcesses(excelRanges):
    # 181, 22, 16
    runExcels = excelRanges

    excelNames = ["Train", "Test", "Validation"]
    excelFinalNames = ["TrainMerged", "TestMerged", "ValidationMerged"]

    counterName = 0

    for r in runExcels:
        excl_list = []
        allCSVs = []

        for file in r:
            fileCSV = excelNames[counterName] + str(file) + ".csv"
            if os.path.exists(fileCSV):
                excl_list.append(pd.read_csv(fileCSV, header=None))
                allCSVs.append(fileCSV)
        excl_merged = pd.concat(excl_list, ignore_index=True)
        excl_merged.to_csv(excelFinalNames[counterName] + ".csv", index=False, chunksize=100000)
        CSVToExcel(excelFinalNames[counterName] + ".csv")
        for csvs in allCSVs:
            if os.path.isfile(csvs):
                os.remove(csvs)
            if os.path.isfile(str(csvs).replace("csv", "xlsx")):
                os.remove(str(csvs).replace("csv", "xlsx"))
        counterName += 1


from SummarizerNLTK import BestSummarizer


def procClipDescr(df, rowList, filename, id):
    counter = 1
    print("Before Final" + str(df.columns))
    for row in rowList:
        prphsed = BestSummarizer(row.name).Paraphrased
        df['2'][int(row.index)] = prphsed
        # print("Id = " + str(id) + " " + str(counter))
        counter += 1
    df.rename(columns={'2': 'input1'}, inplace=True)
    df.rename(columns={'1': '2'}, inplace=True)
    df.rename(columns={'input': '1'}, inplace=True)
    df.rename(columns={'input1': 'input'}, inplace=True)

    df.to_excel(str(filename).replace(".xlsx", "WithSummary.xlsx"), header=df.columns.values, index=0)
    df.to_csv(str(filename).replace(".xlsx", "WithSummary.csv"), header=df.columns.values, index=0)


def GenerateSummary(fileName):
    dataset = pd.read_excel(fileName, header=0)

    clipList = list(dataset["input"].values)
    # print(dataset.columns)
    counter = 0
    newList = []
    for i in clipList:
        newList.append(Row(i, counter))
        counter += 1
    procClipDescr(dataset, newList, fileName, 1)


def excelToCSV(file):
    read = pd.read_excel(file, header=0)
    read.to_csv(str(file).replace(".xlsx", ".csv"), index=False)


def addColumnsToExcel(excelname):
    excel = pd.read_excel(excelname, header=0)

    excel.columns = ['0', 'target', 'input', '1', str("2"), '3']
    excel.rename(columns={2: '2'}, inplace=True)
    excel.to_excel(excelname, header=excel.columns, index=False)
    excel.to_csv(str(excelname).replace("xlsx", "csv"), header=excel.columns, index=False)


def initCLIP():
    global model, clip_model, preprocess, tokenizer
    model, clip_model, preprocess, tokenizer = setMainModels()


def runChatGPT():
    from chatGPTFeed import chatGPTSummary as chatgptsum
    dataset = pd.read_csv("TrainMerged.csv", header=0, sep=",")[:50]
    import time
    for vals in dataset["input"].values:
        print(chatgptsum(vals).result + "***" + vals + "\n")
        time.sleep(25)


if __name__ == "__main__":
    initCLIP()
    # test clip using standard images
    # NNforImages("bat.png")
    # NNforImages("elephant_and_man.png")
    NNforImages("monkey.png")
    # NNforImages("rabbit_owl.png")
    # test finished

    readFromDataset = 70  # if value >= 0 then reads only first N rows else reads everything inside the dataset - default value = -1
    numOfPerGroup = 10  # group value for each process during the multi-processing - default value = 500
    if 0 <= readFromDataset < numOfPerGroup:
        print("Must be readFromDataset >= numOfPerGroup")
        exit(0)
    else:
        noFiles = 0
        if readFromDataset / numOfPerGroup == int(readFromDataset / numOfPerGroup):
            noFiles = int(readFromDataset / numOfPerGroup)
        else:
            noFiles = int(readFromDataset / numOfPerGroup) + 1

        prepareDatasetStep(dconst.test_csv, readFromDataset, dconst.dsExcelTest, numOfPerGroup)
        prepareDatasetStep(dconst.train_csv, readFromDataset, dconst.dsExcelTrain, numOfPerGroup)
        prepareDatasetStep(dconst.val_csv, readFromDataset, dconst.dsExcelVal, numOfPerGroup)
        if readFromDataset != numOfPerGroup:
            CombineResultsMultiProcesses([range(1, noFiles + 1), range(1, noFiles + 1), range(1, noFiles + 1)])

    addColumnsToExcel("ValidationMerged.xlsx")
    addColumnsToExcel("TrainMerged.xlsx")
    addColumnsToExcel("TestMerged.xlsx")

    GenerateSummary("ValidationMerged.xlsx")
    GenerateSummary("TrainMerged.xlsx")
    GenerateSummary("TestMerged.xlsx")

    excelToCSV("TestMerged.xlsx")
    excelToCSV("TrainMerged.xlsx")
    excelToCSV("ValidationMerged.xlsx")

    from FinetunePegasus import FineTunePegasus as ftPeg
    from FinetuneBART import FineTuneBART as ftBART

    ftPeg("PegasusSummary",
          ["TrainMergedWithSummary.csv", "ValidationMergedWithSummary.csv", "TestMergedWithSummary.csv"])
    # trained with summarized clip text as input
    ftPeg("PegasusCLIP", ["TrainMerged.csv", "ValidationMerged.csv", "TestMerged.csv"])
    # trained with clip text as input

    ftBART("BARTSummary",
           ["TrainMergedWithSummary.csv", "ValidationMergedWithSummary.csv", "TestMergedWithSummary.csv"])
    # trained with summarized clip text as input
    ftBART("BARTCLIP", ["TrainMerged.csv", "ValidationMerged.csv", "TestMerged.csv"])
    # trained with clip text as input

    runChatGPT()

    from MetricsFromModels import getRougeFromModels
    from MetricsFromModels import getRougeFromModelPegasus

    getRougeFromModelPegasus("TestMerged.csv", -1, ",")
    getRougeFromModelPegasus("TestMergedWithSummary.csv", -1, ",")
    print(getRougeFromModels("TestMerged.csv", -1, ",", "facebook/bart-large", "bart").result)
    print(getRougeFromModels("TestMergedWithSummary.csv", -1, ",", "facebook/bart-large", "bart").result)
