# IntentClassification
This repo implements BERT for sequence classification to perform intent classification. 

In this project I use pre-trained BERT model to perform intent classification on  
[“An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction”]( https://github.com/clinc/oos-eval ) dataset. 

IntentClassificationNotebook.ipynb is a jupyter notebook and here I describe the process for loading, preprocessing and feature transformation on the data. 
Then, I show how to fine-tune the pre-trained BERT model for this dataset and evaluate it on the test dataset to get close to 97% accuracy. 
I compare different evaluation metrics and discuss some of the improvements that can be made in future.

IntentClassification.py encapsulates the pipeline created in IntentClassificationNotebook.ipynb.

IntentClassificationWithClassNotebook.ipynb imports the IntentClassification.py and uses its class to perform the intent classification using BERT.
