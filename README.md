# Breast Cancer Classification Using CML for CI/CD

I used Continuous Machine Learning (CML) which is an open-source CLI tool for implementing continuous integration & delivery (CI/CD) with a focus on MLOps.

With CML, our Machine Learning pipelines run automatically when we push git commits, also we can get quick results about our ML model in a README.md file available in Github workspace and in our email.
All the workflow ML processes are defined in a (CML) YAML file.

In this example, I am using a Breast Cancer Dataset available in sklearn to classify whether it is a malignant cancer or benign cancer with the help of SVM as a machine learning algorithm.

## Model metrics

Accuracy score = 0.956140350877193
ROC AUC score = 0.9758597883597884
Confusion Matrix = 
[[37  5]
 [ 0 72]]
 
## Model Performance

The relationship between features (first 3 variables)
![](https://asset.cml.dev/7ee33366800f1c7e8032f32fa57b9dc9d886d186?cml=png)

## Heatmap
![](https://asset.cml.dev/02aa9fd83bfa37b666ad336da9cc26356c0ca644?cml=png)

## ROC CURVE
![](https://asset.cml.dev/d89be7a79291ab3bbbdb820645d28ca3a0efad54?cml=png)
 

![CML watermark](https://raw.githubusercontent.com/iterative/cml/master/assets/watermark.svg)
