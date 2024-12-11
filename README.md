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
- The dataset contains customer information from a bank, like age, default status, and loans.
- The goal is to predict if a customer will subscribe to a term deposit (TD).
- The best model used is the Voting Ensembler.

## Scikit-learn Pipeline
- Use Logistic Regression from sklearn and HyperDrive to find the best model.
- Clean the dataset, then split it into 80% for training and 20% for testing.
- Sampler Benefit: Random Parameter Sampling picks different settings (C and max_iter) to find the best accuracy.
- Early Stopping Benefit: Bandit stops training if accuracy doesn’t improve, saving time and resources.

## AutoML
AutoML can handle tasks like missing values, feature engineering, and hyperparameter tuning automatically using various models. Key parameters include task type (e.g., classification/regression), training data, target column, evaluation metric, and cross-validation.
Random Parameter Sampling: Tries different combinations of parameters randomly to find the best accuracy.
Early Stopping: Stops training when accuracy doesn’t improve, saving time and resources.

## Pipeline comparison
- I don’t see much difference between AutoML and HyperDrive in this case. The accuracy for HyperDrive is XX, while for AutoML it is XX.
- The gap is small because the dataset is small, clean, and has variables strongly related to the target. In real-world projects, datasets are more complex, and AutoML might work better if the person using HyperDrive lacks extra dataset knowledge.
- AutoML is great because it tests many models to find the best one, while HyperDrive here only uses Logistic Regression, which may cause bias.
- Benefit of early stopping: Bandit early stopping stops training when accuracy doesn’t improve, saving time and resources.

## Future work
- I want to recreate AutoML for my real-world problem.
- AutoML feels like a black box to me.
- I don’t understand how it recognizes data types, handles data, or creates new features.

## Proof of cluster clean up
Delete the compute cluster
