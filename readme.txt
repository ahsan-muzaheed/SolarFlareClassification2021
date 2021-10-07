
Step-1:

https://mlstuffs.s3.us-west-1.amazonaws.com/E2E_FlareClassification.zip
Use the above url to Download the zip file which contains source file (main.py) and data files(Sampled_inputs.pck ,Sampled_labels.pck)
After downloading finished ,Extract the zip file. 


Step-2:

Now Install python 3.8.x 
We have used:
https://www.python.org/ftp/python/3.8.5/python-3.8.5-amd64.exe
During installation make sure you have “Disabled path length limit”

Step-3:

Open cmd prompt and navigate to the root folder of the downloaded project where “requirements.txt” exists.
Now execute this command:
pip install -r requirements.txt
It will install all the dependencies with the proper version.

Note 1:
For some cases, you might be required to execute this command before proceeding to the earlier cmd:
pip install wheel
If you are using Anaconda or any other kind of python environment where pip cmd is not available then make sure to install the dependencies with the proper version listed in requirements.txt

Step-4 (optional):

You Might want to  tweak these variables in main.py:
test_sizes = [0.3]
HIDDEN_DIMs = [128]
numEpochs = 1

Step-5:
Once all dependencies installed,navigate to the extracted project's root directory where main.py exist.
Then You can lunch the project using cmd:
python main.py


Note 2:
We have tested it in Windows 10 environment.

