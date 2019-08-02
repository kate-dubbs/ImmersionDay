# Immersion Day Lab Guide

This repository will walk you through mutliple Amazon SageMaker labs. These labs are meant to give you a basic overview of Amazon SageMaker and will primarily leverage example notebooks. More advance discussions on how to implement these features will be addressed later in the day during the whiteboarding and experimentation session. The goal of this immersion day is to provide the foundational knowledge on how to use the key features of Amazon SageMake and begin to understand how Amazon SageMake can be leveraged to accelerate the process of building and deploying machine learning models. 

#### Table of Contents
[Lab 1: Amazon SageMaker Notebook Instance Lifecycle Configuration](#lab-1)
[Lab 2: Using the Built-In Random Cut Forest Algorithm](#lab-2) 
[Lab 3: Scikit-Learn Random Forest: Bring your own algorithm and Bring your own container approaches](#lab-3)
[Lab 4: SageMaker PySpark XGBoost Model](#lab-4)

## Lab 1
### Amazon SageMaker Notebook Instance Lifecycle Configuration

Before we dive into the mechanics this workshop, let's launch our first SageMaker notebook and explore some lifecycle configurations we can apply to our notebook. All of these steps can be automated through CloudFormation.

Follow these steps to launch a SageMaker Notebook Instance, download and explore the dataset:
1. Open Amazon SageMaker Console, navigate to ‘Notebook instances‘ under ‘Notebook‘ menu and click on ‘Create notebook instance’. Choose a name for your Notebook instance. For the instance type, leave the default ‘ml.t2.medium’ since our example dataset is small and you won’t use GPUs for running training/inference locally.
For the IAM role, select ‘Create a new role’ and select the options shown below for the role configuration.
 

Click ‘Create role’ to create a new role and then hit ‘Create notebook instance’ to submit the request for a new notebook instance.
![alt text](https://github.com/UPDATE/images/img1.png "Creating a notebook instance")

2. SageMaker Notebooks have feature that allows you to optionally sync the content with a Github repository (ADD GIT REPO FOR LAB). Since you'll be using Notebook file and other files from this repository throughout this workshop, add the URL of this repository to have this cloned onto your instance, upon creation.
![alt text](https://github.com/UPDATE/images/img2.png "Cloning GitHub Repository")

 
Note: It usually takes a few minutes for the notebook instance to become available. Once available, the status of the notebook instance will change from ‘Pending’ to ‘InService’. You can then follow the link to open the Jupyter console on this instance and move on to the next steps.

2a. Alternatively, you can create a Notebook lifecycle configuration, to add the code to clone the Github repository. This approach is particularly useful, if you want to reuse a notebook that you might already have.
Assuming your Notebook instance is in stopped state, add the following code into a new 
Lifecycle configuration, attach the configuration to your notebook, before starting the instance. 

![alt text](https://github.com/UPDATE/images/img3.png "Creating Lifecycle Configuration")
 
```
#!/bin/bash
set -e
cd /home/ec2-user/SageMaker
git clone ADD GIT REPO URL
sudo chown ec2-user:ec2-user -R ADD FOLDER!/
```

![alt text](https://github.com/UPDATE/images/img4.png "Notebook Configuration")	 

With the config attached, when your Notebook instance starts, it will automatically clone this repository.

3. Click 'Open Jupyter' once your notebook is online, if there is time left in the workshop click on the 'SageMaker Examples' tab at the top and work through an example notebook you find interesting.

For more example lifecycle configurations: https://github.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples

For an example on using CloudFormation to automate notebook deployment: https://course.fast.ai/start_sagemaker.html


## Lab 2
### Using the Built-In Random Cut Forest Algorithm

In this section, you’ll work your way through a Jupyter notebook that demonstrates how to use a built-in algorithm in SageMaker. More specifically, you’ll use SageMaker’s Random Cut Forest (RCF) algorithm, an algorithm designed to detect anomalous data points within a dataset. Examples of when anomalies are important to detect include when website activity uncharacteristically spikes, when temperature data diverges from a periodic behavior, or when changes to public transit ridership reflect the occurrence of a special event.

The Amazon SageMaker Random Cut Forest (RCF) algorithm is an unsupervised algorithm for detecting anomalous data points within a dataset. In particular, the RCF algorithm in Amazon SageMaker associates an anomaly score with each data point. An anomaly score with low values indicates that the data point is considered “normal” whereas high values indicate the presence of an anomaly. The definitions of “low” and “high” depend on the application, but common practice suggests that scores beyond three standard deviations from the mean score are considered anomalous.

The RCF algorithm in Amazon SageMaker works by first obtaining a random sample of the training data. Since the training data may be too large to fit on one machine, a technique called reservoir sampling is used to efficiently draw the sample from a stream of data. Subsamples are then distributed to each constituent tree in the random cut forest. Each subsample is organized into a binary tree by randomly subdividing bounding boxes until each leaf represents a bounding box containing a single data point. The anomaly score assigned to an input data point is inversely proportional to its average depth across the forest. 

In this notebook, we will use the SageMaker RCF algorithm to train a model on the [Numenta Anomaly Benchmark (NAB) NYC Taxi dataset](https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv) which records the amount New York City taxi ridership over the course of six months. We will then use this model to predict anomalous events by emitting an “anomaly score” for each data point.

Running the notebook
1.	Access the SageMaker notebook instance you created earlier. Open the [SageMaker Examples](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-nbexamples.html) tab.
2.	In the **Introduction to Amazon Algorithms** section locate the **random_cut_forest.ipynb** notebook and create a copy by clicking on **Use**.
3.	You are now ready to begin the notebook.
4.	Follow the directions in the notebook. The notebook will walk you through the data preparation, training, hosting, and validating the model with Amazon SageMaker. 

## Lab 3
### Scikit-Learn Random Forest: Bring your own algorithm and Bring your own container approaches

In this section, you’ll work your way through two Jupyter notebooks that demonstrate how to use the bring your own algorithm and bring your own container approaches for training models in SageMaker. More specifically, you’ll use Scikit-Learn's Random Forest algorithm, an algorithm designed to detect anomalous data points within a dataset. Both of the model training algorithms will be the same, but they will be deploy in a slightly different manner. These models will be trained using the Boston house prices dataset (https://scikit-learn.org/stable/datasets/index.html#boston-dataset).

Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy to use algorithm. A forest is comprised of trees. It is said that the more trees it has, the more robust a forest is. Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good indicator of the feature importance.

Random forests has a variety of applications, such as recommendation engines, image classification and feature selection. It can be used to classify loyal loan applicants, identify fraudulent activity and predict diseases. It lies at the base of the Boruta algorithm, which selects important features in a dataset.

Running the notebooks
1.	Access the SageMaker notebook instance you created earlier. Open the **SageMaker Examples** tab.
2.	In the **SageMaker Python SDK** section locate the **SKlearn_on_SageMaker_end2end.ipynb** notebook and create a copy by clicking on **Use**.
3.	You are now ready to begin the notebook.
4.	Follow the directions in the notebook. The notebook will walk you through the data preparation, training, hosting, and validating the model with Amazon SageMaker. Once completed, return from the notebook to these instructions.
5.	Once the BYOA notebook is completed, open the **SageMaker Home** tab.
6.	Click on the **SKLearn_Container_RF.ipynb** 
7.	Follow the directions in the notebook. The notebook will walk you through building a container for training, hosting, and validating the model with Amazon SageMaker. 

## Lab 4
### SageMaker PySpark XGBoost Model

In this section, you’ll work your way through your final Jupyter notebook. This notebook will show how to classify handwritten digits using the XGBoost algorithm on Amazon SageMaker through the SageMaker PySpark library. We will train on Amazon SageMaker using XGBoost on the MNIST dataset, host the trained model on Amazon SageMaker, and then make predictions against that hosted model.

Unlike the other notebooks that demonstrate XGBoost on Amazon SageMaker, this notebook uses a SparkSession to manipulate data, and uses the SageMaker Spark library to interact with SageMaker with Spark Estimators and Transformers.
- You can visit SageMaker Spark's GitHub repository at https://github.com/aws/sagemaker-spark to learn more about SageMaker Spark.
- You can visit XGBoost's GitHub repository at https://github.com/dmlc/xgboost to learn more about XGBoost

Running the notebooks
1.	Access the SageMaker notebook instance you created earlier. Open the SageMaker Examples tab.
2.	In the **Sagemaker Spark** section locate the **pyspark_mnist_xgboost.ipynb** notebook and create a copy by clicking on **Use**.
3.	You are now ready to begin the notebook.
4.	Follow the directions in the notebook. The notebook will walk you through the data preparation, training, hosting, and validating the model with Amazon SageMaker. 



