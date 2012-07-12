import utilityFunctions
from utilityFunctions import *

# Classify documents using two classifiers - Linear SVM & Naive Bayes
# and compare the results
def classify_svm_vs_nb():
	# Read and normalize the data
	[x_all,y_all] = readRawData("../data/rawdata.txt")
	x = array(normalize_scale(x_all[1:]))
	y = array(normalize_scale(y_all[1:]))

	# Separate classifier for each label (we call a label in the label set as 'Tag')
	# For now, we don't try to model tag correlations or use multi-label learning techniques
	for tagnum in range(len(y[0])):
		# Extract this tag value for all examples, from the label set
		yl = array([y[i][tagnum] for i in range(len(y))])
		yl_01 = array([y_all[i+1][tagnum] for i in range(len(y))]) # These are the unnormalized 0/1 values
		ystrat = stratifier(yl_01, 2) # Stratifier ensures that each data split has equal proportion of +ve labels

		# Print the tag name
		print str(y_all[0][tagnum])
		
		# Train-test split, 5-times
		sss = StratifiedShuffleSplit(y=ystrat, n_iterations=5, test_size=88)
		for train_index, test_index in sss:
			xtr = x[train_index]   # bag of words - training
			ytrl = yl[train_index] # tag value - training
			xte = x[test_index]    # bag of words - test set
			ytel = yl[test_index]  # true tag value - test set

			# Regularization strength values for performing Grid search
			parameters1 = {'C':[2**i for i in range(-14,14,2)]} # For linear SVM
			parameters2 = {'alpha':[2**i for i in range(-14,14,2)]} # For Naive Bayes
		
			# Initialize the two classifiers
			clf1 = LinearSVC(penalty='l2', loss='l1', scale_C=True)
			clf2 = BernoulliNB(alpha=-1000)
		
			# Do parameter tuning via cross validation within the training set
			classifier_cv1 = GridSearchCV(clf1, parameters1, fit_params={'class_weight':'auto'})
			classifier_cv2 = GridSearchCV(clf2, parameters2) 
			ystrat = stratifier(ytrl,3)
			sc1 = classifier_cv1.fit(xtr, ytrl, cv=StratifiedKFold(ystrat, 3), refit=True, n_jobs=1)
			sc2 = classifier_cv2.fit(xtr, ytrl, cv=StratifiedKFold(ystrat, 3), refit=True, n_jobs=1)

			ytelhat1 = sc1.predict(xte) # ytelhat stands for y_test_label. Hat implies that this is an estimate
			ytelhat2 = sc2.predict(xte)

			[testloss1, testprecision1, testrecall1] = losses(ytel, ytelhat1)
			[testloss2, testprecision2, testrecall2] = losses(ytel, ytelhat2)
			print " Linear SVM:"
			print "  Test Accuracy: " + str(1-testloss1) + "\tPrecision: " + str(testprecision1) + "\tRecall: " + str(testrecall1)
			if (testprecision1 + testrecall1 > 0):
				print "  Test F-Measure: " + str(2*testprecision1*testrecall1/(testprecision1+testrecall1))
			print " NB:"
			print "  Test Accuracy: " + str(1-testloss2) + "\tPrecision: " + str(testprecision2) + "\tRecall: " + str(testrecall2)
			if (testprecision2 + testrecall2 > 0):
				print "  Test F-Measure: " + str(2*testprecision2*testrecall2/(testprecision2+testrecall2))
			
			# Debug Values. Enable to print:
			#print "  labels in training: " + str(Counter(ytrl))
			#print "  labels in test: " + str(Counter(ytel))
			#print "  Training Accuracy: " + str(1-trainingloss) + "\tPrecision: " + str(trainingprecision) + "\tRecall: " + str(trainingrecall)
			#print "  Training predictions: " + str(Counter(ytrlhat))
			#print "  Test Predictions: " + str(Counter(ytelhat))
			#print "  Cross-validation Accuracy: " + str(sc.best_score_)
			#print " Best regularization parameter C: " + str(sc.best_estimator_.C)
			#pdb.set_trace()

# Main Entry Point:
classify_svm_vs_nb()
pdb.set_trace()

