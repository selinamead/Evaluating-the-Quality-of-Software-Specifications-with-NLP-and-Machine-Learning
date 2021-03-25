# Evaluating-the-Quality-of-Software-Specifications-with-NLP-and-Machine-Learning

  ```
  Selina Mead (Miller)
  University of Tuebingen
  4065083
  selinamead@live.com
  
  ```

## Contents
* Code to Classify Software Specifications (SS) as acceptable or unacceptable
* Dataset containing 2500 German textual SS
* Thesis paper (to be completed)


### Code

Classification_and_Evaluation.py
```
This is the main python class and implements Classification models
```
NLP_features.py
```
This class generates Natural Language features such as:
  - No. of words
  - Syllables per word
  - Internal Punctuation
  - Flesch Index
  - ...
```
Utils.py
```
This is a helper class for mapping dataframes
```

* Details of how to run the program will be provided at a later date



### Dataset

2500 actual German textual software specifications manually evaluated and rated by industry professionals. Each SS has a rating between 1 - 5, with 1 being the worst and 5 the best rating. 
* Example of a SS
```
Der maximal zulässige Kalibrierfehler des Nickwinkels beträgt ±0,15° = 5
```
* More information on the dataset will be provided at a later date

### Thesis Paper

* Details of paper will be added at a later date

