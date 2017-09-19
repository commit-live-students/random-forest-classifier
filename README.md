# Random Forest Classifier

We have learnt about both decision trees and random forest. So the question is why use Random Forest, if Decision Trees serve the purpose and are more interpretable?

Let's explore the same in this assignment.

### Task 1: Write a function called myRandomForestClassifier():

- Accepts the following parameters:
    * X_train, y_train, X_test (Numpy arrays for training, testing; any format acceptable by sklearn will work)
    * paramgrid (list of parameters (including those of the classfier) for GridSearchCV)
    * KFold (the number of k-folds to be used in cross-validation) (Optional) (Default 3)
    * early_stopping_rounds (Int) (Optional) (Default 10)
    * seed (a number; a subsequent call to the function with the same seed will reproduce the same results)(optional) (Default 42)
    * **kwargs (To set parameters to the base classifier)

- Should return
    * predictions for X_test
    * trained GridSerchCV object

### Task 2: Figure out the best parameters for Pima Indian Diabetes Dataset

- Write a function called finetune_class which
    * Takes in X_train, X_test, y_train, param_grid
    * Returns y_pred_test, Trained GridSearch Object

- You will use myRandomForestClassifier() function
- Based on the stage-wise optimization values further fine tune the model
- You will need to provide a parameter grid to the said function, be careful to choose which values to be optimized.

