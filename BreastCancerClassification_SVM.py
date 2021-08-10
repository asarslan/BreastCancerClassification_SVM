import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
	
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
	
data = load_breast_cancer() # loading cancer data from the Sklearn library
df = pd.DataFrame(data.data, columns=data.feature_names) # creating dataframe from the features
df['target'] = data.target # adding the target attribute into the dataframe
print(df.head())
	
sns.pairplot(df, hue= 'target', vars= ['mean radius', 'mean texture', 'mean perimeter']) # visualize the relationship between the first 3 features
plt.show()
plt.savefig('pairplot.png', dpi=120)
	
plt.figure(figsize=(25,12))
sns.heatmap(df.corr(), annot=True)
plt.show()
plt.savefig('heatmap.png', dpi=120)
	
X = df.drop(['target'], axis= 1) # features
print(X.head())
	
y = df['target'] # target
print(y.head())
	
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1) # creating the training and testing data
print("X_train size is", X_train.shape)
print("X_test size is", X_test.shape)
print("y_train size is", y_train.shape)
print("y_test size is", y_test.shape)
	
svc_clf = SVC(kernel= "linear",probability= True) #creating SVC model with linear kernel
svc_clf.fit(X_train, y_train) # training SVC model
	
print("Confusion matrix: " + '\n' + str(confusion_matrix(y_test, svc_clf.predict(X_test)))) #confusion matrix
	
print("Accuracy score: " + str(accuracy_score(y_test, svc_clf.predict(X_test)))) #accuracy score
	
print("ROC AUC score: " + str(roc_auc_score(y_test, svc_clf.predict_proba(X_test)[:, 1]))) # ROC (AUC) score
	
#plotting the SVC model’s ROC curve
fpr, tpr, thresholds = roc_curve(y_test, svc_clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label= 'ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label= 'Random guess')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.legend(loc= "lower right")
#plt.show()
plt.savefig('model_results.png', dpi=120)

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nAccuracy score = {round(accuracy_score(y_test, svc_clf.predict(X_test)), 2)}')
    outfile.write(f'\nROC AUC score = {round(roc_auc_score(y_test, svc_clf.predict_proba(X_test)[:, 1]), 2)}')
    outfile.write(f'\nConfusion Matrix = {round(confusion_matrix(y_test, svc_clf.predict(X_test)), 2)}')
