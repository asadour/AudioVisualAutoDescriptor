# AudioVisualAutoDescriptor

# 1) In order to install CLIP please use the following command in pycharm's command line:
#  pip install git+https://github.com/openai/CLIP.git
# 2) Download clip model from https://www.dropbox.com/scl/fi/tguafg8z7nnrf840t9ycd/conceptual_weights.pt?rlkey=un323lkusap0t4mgnsg3llviv&dl=0 
# and place it inside the project (pytorch file: conceptual_weights.pt)
# 3) As well as install openpyxl by using command:  pip install openpyxl
# and 	pip install protobuf  
# 4) pip install rouge_score
## !ATTENTION! ##

* Default value for readFromDataset variable (in main.py) is -1 or any negative integer number.
* Default value for numOfPerGroup variable (in main.py) is 500.

Used python 3.10

Upon starting the execution of the code we test the CLIP as following:
  !!TEST CLIP:!! a chimpanzee with a finger in its mouth.

![monkey](https://github.com/asadour/AudioVisualAutoDescriptor/assets/22840678/0303167f-6421-4518-b14a-e8143a837927)


That's it...
