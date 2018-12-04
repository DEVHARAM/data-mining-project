from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from FFP.module import ffp, initial, frequency
import pickle
import os


def preprocess(path, p):
	count = 0
	indexs = frequency.convert_index(path, "public/first.txt", p)

	with open("public/first.txt", 'r', encoding='utf8') as f:
		read = f.read()
		for index in indexs:
			read = read.replace(index, initial.initial(index))

	with open("public/second.txt", 'w', encoding='utf8') as f:
		for line in read.split('\n'):
			if line != '\n':
				f.write(line+'\n')

	with open("public/second.txt", 'r', encoding='utf8') as In:
		with open("public/third.txt", 'w', encoding='utf8') as Out:
			for line in iter(lambda: In.readline(), ''):
				if line[0] == '2' or line[0] == '0':
					print(line)
					count += 1
					Out.write(line)

	train_num = int(count*0.8)
	test_num = count-train_num

	with open("public/third.txt", 'r', encoding='utf8') as In:
		with open("public/train.txt", 'w', encoding='utf8') as train:
			for i in range(train_num):
				train.write(In.readline())

		with open("public/test.txt", 'w', encoding='utf8') as test:
			for i in range(test_num):
				test.write(In.readline())


def work(train_path, test_path, k):

	features = ffp.parsing(train_path, k)
	train = ffp.matrix(train_path, features, k)

	train_data = train[:, 0:len(features)-1]
	train_label = train[:, len(features)-1]

	test = ffp.matrix(test_path, features, k)

	test_data = test[:, 0:len(features)-1]
	test_label = test[:, len(features)-1]
	gamma_range = [0.01, 0.1, 1.0, 10.0]

	parameter_grid = [
				{'gamma': gamma_range, 'kernel': ['rbf']},
				]
	grid = GridSearchCV(SVC(), parameter_grid, scoring='accuracy', cv=5)
	grid.fit(train_data, train_label)
	print('best params:', grid.best_params_)

	# insert best params to test
	clf = SVC(**grid.best_params_)
	clf = clf.fit(train_data, train_label)
	
	filename = 'save_model.sav'
	pickle.dump(clf, open(filename, 'wb'))

	pred = clf.predict(train_data)
	print("Train k :"+str(k)+" = "+str(accuracy_score(train_label, pred)))

	print(classification_report(train_label, pred, target_names=['class 0', 'class 1']))

	pred = clf.predict(test_data)
	print("Test k :"+str(k)+" = "+str(accuracy_score(test_label, pred)))

	print(classification_report(test_label, pred, target_names=['class 0', 'class 1']))
	return accuracy_score(test_label, pred)


prepro = os.path.abspath(__file__ + "/../../") + "/prepro/simple.txt"
preprocess(prepro, 1)

work("public/train.txt", "public/test.txt", 2)

"""
p = mp.Pool(3)
k_range=list(range(2,5))
value=[]

value=p.map(work,k_range)

plt.title("Bad Comment(FFP)")
kp=["k2","k3","k4"]
plt.plot(kp, value,'bs--')
plt.grid()
plt.show()
"""
