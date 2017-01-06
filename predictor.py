import nltk
#import cPickle
from LoadData import readFromFile
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist

import time



class EthnicityPredictor():
	

	def get3_let(self, name):
		return [name[i:i+3] for i in range(len(name) - 3)]

	def buildFeatureList(self, data):
		features = set()
		
		for (name, stat, ethList) in data:
			x_lets = self.get3_let(name)
			for x_let in x_lets:
				features.add(x_let)

		self._featureList = sorted(features) #sorted(features.iteritems(), key=lambda (w,s): w, reverse=True) 
		self._featureNumber = len(self._featureList)
		print(self._featureNumber)

	def getFeature(self, name):
		initVal = [False] * self._featureNumber
		feature = dict(zip(self._featureList, initVal))
		x_lets = self.get3_let(name)
		for x_let in x_lets:
			feature[x_let] = True

		return feature

	
	def getFeatureWithLable(self, data):
		print(data[0])
		print(len(data))

		self.buildFeatureList(data)

		train_data = []

		cnt = 0;
		for (name, stat, ethList) in data:
			if stat < 100000:
				break

			cnt += 1
			if (cnt == 20):
				cnt = 0
				print(cnt, stat)
			
			for i in range(5):
				ethic = self.ethicity[i]
				num = int(ethList[i] / 1000)
				for j in range(num):
					train_data.append((self.getFeature(name), ethic))

		return train_data



	def CreatNaiveBayes(data):
		

		label_freqdist = FreqDist()
		for (name, total, ethList) in data:
			for i in range(len(ethicity)):
				label_freqdist[ethicity[i]] += ethList[i]

		label_probdist = ELEProbDist(label_freqdist)

		feature_freqdist = defaultdict(FreqDist)
		#for (name, total, ethList) in data:

		#	x-lets


		feature_probdist = {}
		for ((label, fname), freqdist) in feature_freqdist.items():
			probdist = estimator(freqdist, bins=len(feature_values[fname]))
			feature_probdist[label, fname] = probdist
		
		nltk.NaiveBayesClassifier = (label_probdist, feature_probdist)




	
	def TrainAndTest(self):
		self._fileData = readFromFile("surname_ethnicity_data.csv")
		self.ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']

		train_set = self.getFeatureWithLable(self._fileData)
		print("total :", len(train_set))

		time2 = time.time()
		print(time2 - time1)


		self.train(train_set)

		time3 = time.time()
		print(time3 - time2)

		file_train_result = open('data.pkl', 'wb')
		pickle.dump(self.classifier, file_train_result, -1)

		time4 = time.time()
		print(time4 - time3)

	def train(self, train_set):
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def classify(self, name):
		feature = getFeature(name)
		return self.classifier.classify(feature)

	def test(self,test_set):
		return nltk.classify.accuracy(self.classifier, test_set)

		
time1 = time.time()

predictor = EthnicityPredictor()
predictor.TrainAndTest()


time5 = time.time()
print(time5 - time1)