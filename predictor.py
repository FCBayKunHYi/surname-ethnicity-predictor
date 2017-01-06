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

		for (name, stat, ethList) in data:
			if stat < 10000:
				break
			for i in range(5):
				ethic = self.ethicity[i]
				num = int(ethList[i] / 100)
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




	@profile
	def TrainAndTest(self):
		self._fileData = readFromFile("surname_ethnicity_data.csv")
		self.ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']

		train_set = self.getFeatureWithLable(self._fileData)
		#self.train(train_set)

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


time2 = time.time()
print(time2 - time1)