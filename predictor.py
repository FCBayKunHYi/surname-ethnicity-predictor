import nltk
import pickle
from LoadData import readFromFile
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist

import time



class EthnicityPredictor():
	
	def __init__(self):
		self._ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']

	def get3_let(self, name):
		return [name[i:i+3] for i in range(len(name) - 3)]

	def buildFeatureList(self, data):
		features = dict()
		
		for (name, stat, ethList) in data:
			if stat < 10000:
				break
			x_lets = self.get3_let(name)
			for x_let in x_lets:
				if x_let in features.keys():
					features[x_let] += stat
				else:
					features[x_let] = stat

		self._featureFdist = features #sorted(features.iteritems(), key=lambda (w,s): w, reverse=True) 	
		#number = [0, 1, 2, 3, 4]
		#self._mapEthic2Num = dict{zip(self._featureList, number)}
		self._featureNumber = len(self._featureFdist)
		print(self._featureNumber)

	
	def getFeature(self, name):
		#initVal = [False] * self._featureNumber
		feature = self._featureFdist  #dict(zip(self._featureList, initVal))
		for label in feature:
			feature[label] = False
		x_lets = self.get3_let(name)
		for x_let in x_lets:
			feature[x_let] = True

		return feature
	
	'''
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
				ethic = self._ethicity[i]
				num = int(ethList[i] / 1000)
				for j in range(num):
					train_data.append((self.getFeature(name), ethic))

		return train_data
	'''
	def saveTrainingResult2pkl(self, filename):
		file_train_result = open(filename, 'wb')
		pickle.dump(self.classifier, file_train_result, -1)

	def readProbabilityFromPkl(self, filename):
		pkl_file = open(filename, 'rb')
		self.classifier = pickle.load(pkl_file)


	def CreatNaiveBayes(data):
		

		label_freqdist = FreqDist()
		for (name, total, ethList) in data:
			for i in range(5):
				label_freqdist[self._ethicity[i]] += ethList[i]

		label_probdist = ELEProbDist(label_freqdist)

		feature_freqdist = defaultdict(FreqDist)
		#for (name, total, ethList) in data:

		#	x-lets
		for (name, total, ethList) in data:
			x_lets = get3_let(name)
			for i in range(5):
				for x_let in x_lets:
					feature_freqdist[(self._ethicity[i], x_let)][True] += ethList[i]
		for ((label, x_lets), freqdist) in feature_freqdist.items():
			num = 0
			for i in range(5):
				if label == self._ethicity[i]:
					num = i
					break
			tot = 0
			for (name, total, ethList) in data:
				if x_let not in name:
					tot += ethList[num]
			feature_freqdist[(label, x_lets)][None] += tot;

		feature_probdist = {}
		for ((label, fname), freqdist) in feature_freqdist.items():
			probdist = ELEProbDist(freqdist, bins=len(self._featureFdist[fname]))
			feature_probdist[label, fname] = probdist
		
		nltk.NaiveBayesClassifier = (label_probdist, feature_probdist)



	
	def TrainAndTest(self):
		self._fileData = readFromFile("surname_ethnicity_data.csv")
		train_set = self._fileData[:300]
		self.buildFeatureList(train_set)
		self.CreatNaiveBayes(train_set)
		time2 = time.time()
		print(time2 - time1)

		self.saveTrainingResult2pkl('train_result.pkl')

		'''
		train_set = self.getFeatureWithLable(self._fileData)
		print("total :", len(train_set))

		time2 = time.time()
		print(time2 - time1)


		self.train(train_set)

		time3 = time.time()
		print(time3 - time2)

		file_train_result = open('train_result.pkl', 'wb')
		pickle.dump(self.classifier, file_train_result, -1)

		time4 = time.time()
		print(time4 - time3)
		'''

	def train(self, train_set):
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def classify(self, name):
		feature = self.getFeature(name)
		print(self.classifier.classify(feature))
		score = self.classifier.prob_classify(feature)
		#score = score.items()
		print('Probability:')
		for ethic in self._ethicity:
			print(ethic, ': ', score.prob(ethic))


	def test(self,test_set):
		return nltk.classify.accuracy(self.classifier, test_set)

		
time1 = time.time()

predictor = EthnicityPredictor()
#predictor.readProbabilityFromPkl('some_train_result.pkl')
predictor.TrainAndTest()
time5 = time.time()
print(time5 - time1)

surname = input("Please input surname:\n")
predictor.classify(surname)
