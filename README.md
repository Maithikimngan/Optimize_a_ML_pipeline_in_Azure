# Optimizing an ML Pipeline in Azure

## Overview
This is a project from the Udacity Azure ML Nanodegree.
We create and improve an Azure ML pipeline using Python SDK and a Scikit-learn model.
The model is then compared with results from Azure AutoML.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
- The dataset includes  customer information in banking, like age, default status, and loans and etc.
- The objective is to predict if a customer will subscribe to a term deposit.
- The best model used is the Voting Ensembler. Because it combines the predictions of multiple models to improve accuracy and robustness

## Scikit-learn Pipeline
- Use Logistic Regression from sklearn and HyperDrive to find the best model.
- First, we need to clean the dataset.
- Second, we split it into 80% for training and 20% for testing Sampler
- The benefit is Random Parameter Sampling picks different settings (C and max_iter) to find the best accuracy.
- And the benefit of Early Stopping is Bandit stops training if accuracy doesn’t improve, saving time and resources.

## AutoML
AutoML is a model that can do missing values, feature engineering, and hyperparameter tuning automatically using various models. It can also do task like cross-validation to get the best parameter and can try many types of model to find the best model for the task.
The benefit of Early Stopping is the model will stop training when accuracy doesn’t improve, saving time and resources.

## Pipeline comparison
HyperDrive Accuracy
![image](https://github.com/user-attachments/assets/a2921493-2b79-4323-aeab-08442cfba1e7)


AutoML Accuracy
![image](https://github.com/user-attachments/assets/e8078d2c-c4a5-4148-b702-d5e0fcc38a75)



- The accuracy between AutoML and HyperDrive likely the same. The accuracy for HyperDrive is 90.8%, while for AutoML it is 91.9%.
- In real business use case the step of feature engineering is very important. And AutoML can create scripts and test with features will be important for model.
- In this lab, the input feature seems to be very related to output target so that's why feature engineering in this case is not so much important. That's why the two models HyperDrive and AutomML is very close.

## Future work
HyperDrive we can try different set of C, max_iter in Random Parameter Sampling, and can try to increase the max_total_run and max_concurrent_runs to see if there is any accuracy improvement.

## Proof of cluster clean up
Delete the compute cluster


![image](https://github.com/user-attachments/assets/9ee36881-9c56-41e1-9bf1-5010c4c6a72d)



![image](https://github.com/user-attachments/assets/bed8f42e-f7d0-410e-8bd6-40d397789085)


