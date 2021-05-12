# IMPORTING EVERY REQUIRED LIBRARY FIRST
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#SOME SKLEARN TOO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#FOR IGNORING WARNING
import warnings
warnings.filterwarnings('ignore')


#READING THE CSV FILE
df = pd.read_csv('C:/Users/Aseem Rajvanshi/Desktop/work/heart_disease.csv', encoding='cp1252')

#SOME DESCRIPTION
df.head()
df.info()

#HEATMAP WITH COOL COLOURS
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, cmap='cool')
plt.show()

#BAR GRAPH 
sns.countplot(df.target, palette=['green', 'red'])
plt.title("[0] == Not Disease, [1] == Disease");

#FREQUENCY OF DISEASE AS THE AGE GOES
plt.figure(figsize=(15, 8))
sns.countplot(x='age', hue='target', data=df, palette=['#1CA53B', 'red'])
plt.legend(["Haven't Disease", "Have Disease"])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#FREQUENCY OF DISEASE W.R.T. VARIOUS ASPECTS
fig, axes = plt.subplots(3, 2, figsize=(12,12))
fs = ['cp', 'fbs', 'restecg','exang', 'slope', 'ca']
for i, axi in enumerate(axes.flat):
    sns.countplot(x=fs[i], hue='target', data=df, palette='bwr', ax=axi) 
    axi.set(ylabel='Frequency')
    axi.legend(["Haven't Disease", "Have Disease"])
    
    
#SCATTER PLOT BETWEEN 'RESTING BLOOD PRESSURE' AND 'MAXIMUM HEART RATE ACHIEVED'
plt.figure(figsize=(10,8))
sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
plt.show()

#SCATTER PLOT BETWEEN 'SERUM CHOLESTROL' AND 'MAXIMUM HEART RATE ACHIEVED'
plt.figure(figsize=(10,8))
sns.scatterplot(x='chol',y='thalach',data=df,hue='target')
plt.show()

#THIS SCATTER PLOT SHOWS 'MAXIMUM HEART RATE' FOR VARIOUS 'AGE'GROUP FOR NORMAL AND DISEASED PERSON
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


#PREPARING AND TRAINING THE MODEL
X = df.drop(['target'], axis=1).values
y = df['target'].values

scale = StandardScaler()
X = scale.fit_transform(X)


class Model:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
    def model_str(self):
        return str(self.model.__class__.__name__)
    
    def crossValScore(self, cv=5):
        print(self.model_str() + "\n" + "="*60)
        scores = ["accuracy", "precision", "recall", "roc_auc"]
        for score in scores:  
            cv_acc = cross_val_score(self.model, 
                                     self.X_train, 
                                     self.y_train, 
                                     cv=cv, 
                                     scoring=score).mean()
            
            print("Model " + score + " : " + "%.3f" % cv_acc)
        
    def accuracy(self):
        accuarcy = accuracy_score(self.y_test, self.y_pred)
        print(self.model_str() + " Model " + "Accuracy is: ")
        return accuarcy
        
    def confusionMatrix(self):        
        plt.figure(figsize=(6, 6))
        mat = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(mat.T, square=True, 
                    annot=True, 
                    cbar=False, 
                    xticklabels=["Haven't Disease", "Have Disease"], 
                    yticklabels=["Haven't Disease", "Have Disease"])
        
        plt.title(self.model_str() + " Confusion Matrix")
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values');
        plt.show();
        
    def classificationReport(self):
        print(self.model_str() + " Classification Report" + "\n" + "="*60)
        print(classification_report(self.y_test, 
                                    self.y_pred, 
                                    target_names=['Non Disease', 'Disease']))
    
    def rocCurve(self):
        y_prob = self.model.predict_proba(self.X_test)[:,1]
        fpr, tpr, thr = roc_curve(self.y_test, y_prob)
        lw = 2
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 
                 color='darkorange', 
                 lw=lw, 
                 label="Curve Area = %0.3f" % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='green', 
                 lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.model_str() + ' Receiver Operating Characteristic Plot')
        plt.legend(loc="lower right")
        plt.show()


#________________________________________________________________________________
        
#CLASSIFYING USING RANDOM FOREST CLASSIFIER
rfc = Model(model=RandomForestClassifier(n_estimators=1000), X=X, y=y)
rfc.crossValScore(cv=10)
rfc.accuracy()
rfc.confusionMatrix()
rfc.classificationReport()
rfc.rocCurve()

#________________________________________________________________________________

#CLASSIFYING USING SUPPORT VECTOR CLASSIFIER
svm = Model(model=SVC(C=5, probability=True), X=X, y=y)
svm.crossValScore(cv=10)
svm.accuracy()
svm.confusionMatrix()
svm.classificationReport()
svm.rocCurve()

#________________________________________________________________________________

#CLASSIFYING USING PIPELINE CLASSIFIER
lr = LogisticRegression(C=10, n_jobs=-1)
pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), lr)
pip = Model(model=pipeline, X=X, y=y)
pip.crossValScore()
pip.accuracy()
pip.confusionMatrix()
pip.classificationReport()
pip.rocCurve()

#_________________________________________________________________________________

#CLASSIFYING USING K-NEAREST NEIGHBOR(KNN) CLASSIFIER
knn = Model(model=KNeighborsClassifier(n_neighbors=100), X=X, y=y)

knn.crossValScore()
knn.accuracy()
knn.confusionMatrix()
knn.classificationReport()
knn.rocCurve()

#_________________________________________________________________________________


#HERE WE COMPARE ALL THE CLASSIFIER TECHNIQUES AND FORM A BAR GRAPH
models = [rfc, svm, pip, knn]
names = []
accs = []
for model in models:
    accs.append(model.accuracy())
    names.append(model.model_str())
    
#NOW WE PLOT IT
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1.2,0.1))
plt.ylabel("Accuracy")
plt.xlabel("Algorithms")
sns.barplot(x=names, y=accs)
plt.savefig('models_accuracy.png')
plt.show()    