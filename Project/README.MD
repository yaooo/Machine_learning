# Machine Learning Project
## Prerequisites:
Python 3 required.
Additional python pakages have to be installed based on the requirements.


## Run the code:
* To check the running history (or re-run the code):
```
cd <project directory>
jupyter notebook 
// Then click on the .ipynb files in the local server
```

* To directly run .py code:
```
cd <project directory>
python <code>.py
```

## Code Description:
* white_blood_cells.ipynb: <br>
The playground for building up the network and image/data pre-analysis.

* Training.py: <br>
The source code of the training model. (Note: Due to the initialization of weights, dropout layer and image_generator, it might take a couple of tries in order to get a good model trained.)


* testing_model.py: <br>
  1. To load the model.
  2. To get the validation loss and validation accuracy for the testing and training dataset.
  3. To output the confusion matrix.

* testing_model.ipynb: <br>
The running history of "testing_model.py".


## The Trained Models:
* acc_95_83.h5---------->> 95% training accuracy, 83% validation accuracy
* acc_90_83.h5 --------->> 90% training accuracy, 83% validation accuracy
* acc_87_80.h5 --------->> 87% training accuracy, 80% validation accuracy


## Acknowledgements:
https://github.com/Shenggan/BCCD_Dataset MIT License


## Dataset:
1. https://www.kaggle.com/paultimothymooney/blood-cells
2. https://github.com/Shenggan/BCCD_Dataset


## Author
* @yaooo
