import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
import sys
sys.path.append('module')
import ffp

def load(train_path,test_path,k):
	 features=ffp.parsing(train_path,k)
	 train=ffp.matrix(train_path,features,k)

	 train_data=train[:,0:len(features)-1]
	 train_label=train[:,len(features)-1]

	 test=ffp.matrix(test_path,features,k)

	 test_data=test[:,0:len(features)-1]
	 test_label=test[:,len(features)-1]
	 
	 loaded_model = pickle.load(open("save_model.sav", 'rb'))

	 pred=loaded_model.predict(test_data)

	 print("Test k :"+str(k)+" = "+str(accuracy_score(test_label,pred)))

	 print(classification_report(test_label, pred, target_names=['class 0','class 2']))
	 return accuracy_score(test_label,pred)

load("public/train.txt","public/test.txt",2)
