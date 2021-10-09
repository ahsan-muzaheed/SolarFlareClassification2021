# MVTS-based Solar Flare Classification using LSTM
This is the implementation of the paper: [Sequence Model-based End-to-End Solar FlareClassification from Multivariate Time Series Data](https://www.researchgate.net/publication/355158881_Sequence_Model-based_End-to-End_Solar_Flare_Classification_from_Multivariate_Time_Series_Data)

## DataSet
### Multivariate time series datasets

Dataset Included in the repo.

Sampled_inputs.pck

Sampled_labels.pck


## Instructions
Step-1:

Install python 3.8.x 
We have used:
https://www.python.org/ftp/python/3.8.5/python-3.8.5-amd64.exe
During installation make sure you have “Disabled path length limit”

Step-2:

Open cmd prompt and navigate to the root folder of the downloaded project where “requirements.txt” exists.
Now execute this command:
pip install -r requirements.txt
It will install all the dependencies with the proper version.

Note 1:
For some cases, you might be required to execute this command before proceeding to the earlier cmd:
pip install wheel
If you are using Anaconda or any other kind of python environment where pip cmd is not available then make sure to install the dependencies with the proper version listed in requirements.txt

Step-3 (optional):

You Might want to  tweak these variables in main.py:
test_sizes = [0.3]
HIDDEN_DIMs = [128]
numEpochs = 1

Step-4:
Once all dependencies installed,navigate to the extracted project's root directory where main.py exist.
Then You can lunch the project using cmd:
python main.py


Note 2:
We have tested it in Windows 10 environment.




## Citation

```
@inproceedings{Muzaheed2021ICMLA,
  title={ Sequence Model-based End-to-end Solar Flare Classification from Multivariate Time Series Data},
  author={Muzaheed,A. A. M., Hamdi,S. M. and Boubrahimii,S. F.},
  booktitle={Proceedings of the 2021 IEEE 20th International Conference on Machine Learning and Applications (ICMLA)},
  year={2021}
}
```


