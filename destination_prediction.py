import numpy as np
import user_reader as ur
from sklearn.preprocessing import LabelBinarizer
# from sklearn import naive_bayes as nb
from sklearn import linear_model as lm
# from sklearn import svm
# from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns


def data_prep(dataset):
	X = []
	Y = []
	last_trip = None
	for i in xrange(len(dataset)):
		curr_trip = dataset[i]
		# print curr_trip
		dow = (curr_trip[0] - 11688) % 7

		hod = (curr_trip[1]/60) % 24
		# how = dow * 24 + hod
		# X.append((str(how), item[2]))

		# bi_hod = (item[1]/120) % 12
		# bi_how = dow * 12 + bi_hod
		# X.append((str(bi_how), item[2]))

		# tri_hod = (item[1]/180) % 8
		# tri_how = dow * 8 + tri_hod
		# X.append((str(tri_how), item[2]))

		if i > 0 and curr_trip[0] == last_trip[0]:
			X.append((dow, curr_trip[2], last_trip[2]))
		else:
			X.append((dow, curr_trip[2], 'first'))
		# X.append(('aaa', 'bbb'))
		Y.append(curr_trip[4])
		# print X[-1], Y[-1]
		last_trip = curr_trip
	X = np.array(X)
	Y = np.array(Y)
	return X, Y


def splitDataset(dataset, split_date, min_train_period=8, test_period=1, min_train_size=5):
	start = max(11688, split_date - min_train_period * 7)
	end = split_date + test_period * 7
	train_set = []
	test_set = []
	train_days = []
	split_index = 0
	for i, item in enumerate(dataset):
		if item[0] >= end:
			break
		elif item[0] >= start:
			if item[0] < split_date:
				train_set.append(item)
				if item[0] not in train_days:
					train_days.append(item[0])
			else:
				test_set.append(item)
				if split_index == 0:
					split_index = i
	diff_size = len(train_days) - min_train_size
	if diff_size < 0:
		for j in reversed(xrange(split_index)):
			if dataset[j][0] not in train_days:
				train_days.insert(0, dataset[j][0])
			if len(train_days) > min_train_size or train_days[0] < start:
				break
			else:
				train_set.insert(0, dataset[j])
	if len(train_days) < min_train_size:
		train_set = []
	return train_set, test_set


def predict_eval(test_Y, pred_Y):
	r = 0
	w = 0
	for i in xrange(len(test_Y)):
		if test_Y[i] == pred_Y[i]:
			r += 1
		else:
			w += 1
	return r, w


def feature_extraction(trainX, testX):
	X1, features = train_feature_extraction(trainX)
	X2 = test_feature_extraction(testX, features)
	return X1, X2


def train_feature_extraction(X):
	n, k = X.shape
	for i in xrange(k):
		lbin = LabelBinarizer()
		transformed_X = lbin.fit_transform(X[:, i])
		classes = []
		for c in lbin.classes_:
			classes.append(str(i)+'_'+str(c))
		classes = np.array(classes)
		if transformed_X.shape[1] == 1:
			if len(lbin.classes_) > 1:
				transformed_X = np.concatenate((1 - transformed_X, transformed_X), axis=1)
		if i == 0:
			new_X = transformed_X
			features = classes
		else:
			new_X = np.concatenate((new_X, transformed_X), axis=1)
			features = np.concatenate((features, classes), axis=1)
	# print new_X.shape, features.shape
	return new_X, features


def test_feature_extraction(X, features):
	n, d = X.shape
	n_features = len(features)
	new_X = np.zeros((n, n_features))
	for i in xrange(n):
		for k in xrange(d):
			for f in xrange(n_features):
				if str(k)+'_'+str(X[i, k]) == features[f]:
					new_X[i, f] = 1
					break
	return new_X


def predict_eval_by_time(testX, testY, predY):
	dowR = np.zeros(7)
	dowW = np.zeros(7)
	hodR = np.zeros(24)
	hodW = np.zeros(24)
	for i in xrange(len(testX)):
		trip = testX[i]
		dow = int(trip[0])
		hod = int(trip[1])
		if testY[i] == predY[i]:
			dowR[dow] += 1
			hodR[hod] += 1
		else:
			dowW[dow] += 1
			hodW[hod] += 1
	return dowR, dowW, hodR, hodW


def bar_plot(X1, X2):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	N = len(X1)
	ind = np.arange(N)
	width = 0.35

	rects1 = ax.bar(ind, X1, width, color='green')
	rects2 = ax.bar(ind+width, X2, width, color='red')
	ax.set_xlim(-width, len(ind) + width)
	ax.set_ylabel('Frequency')
	if N == 7:
		ax.set_title('Prediction Performance by Day of Week')
		xTickMarks = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
	elif N == 24:
		ax.set_title('Prediction Performance by Hour of Day')
		xTickMarks = range(0, 24, 1)
	else:
		ax.set_title('Prediction Performance over Time of Year')
		xTickMarks = range(1, N+1, 1)
	ax.set_xticks(ind+width)
	xtickNames = ax.set_xticklabels(xTickMarks)
	plt.setp(xtickNames, rotation=45, fontsize=10)
	ax.legend((rects1[0], rects2[0]), ('Correct Pred', 'Incorrect Pred'))
	plt.show()


def dist_plot(X):
	sns.distplot(X, kde=False, bins=20)
	plt.xlabel('Prediction Performance')
	plt.ylabel('Number of Users')
	plt.title('Distribution of Users by Predictability')
	plt.show()


def scatter_plot(X1, X2):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X1, X2, color='blue', s=5, edgecolor='none')
	ax.set_xlim([0, 200])
	ax.set_ylim([0.0, 1.0])
	ax.set_xlabel('Average Training Size')
	ax.set_ylabel('Prediction Accuracy')
	ax.set_title('Correlation between Usage Frequency and Predictability')
	plt.show()


def predict_over_time():
	users = ur.readPanelData("/Users/zhanzhao/Dropbox (MIT)/TfL/Data/sampleData.csv")
	start = 11688
	end = 12419
	# predictability = []
	dowR = np.zeros(7)
	dowW = np.zeros(7)
	# hodR = np.zeros(24)
	# hodW = np.zeros(24)
	# weekR = np.zeros(105)
	# weekW = np.zeros(105)
	days = range(start, end, 7)
	total_R = 0
	total_W = 0
	# train_R = 0
	# train_W = 0
	for t in days:
		for user in users:
			# day1 = max(start, (t-7*training_period))
			# day2 = t
			# day3 = t + 7
			train, test = splitDataset(user.tripList, t)
			if len(train) > 0 and len(test) > 0:
				train_X, train_Y = data_prep(train)
				test_X, test_Y = data_prep(test)
				if len(np.unique(train_Y)) > 1:
					new_train_X, new_test_X = feature_extraction(train_X, test_X)
					# clf = nb.MultinomialNB()
					clf = lm.LogisticRegression()
					# clf = tree.DecisionTreeClassifier()
					# clf = svm.LinearSVC()
					clf.fit(new_train_X, train_Y)
					pred_Y = clf.predict(new_test_X)
				else:
					pred_Y = [train_Y[0]] * len(test_Y)
				r, w = predict_eval(test_Y, pred_Y)
				total_R += r
				total_W += w

				# new_train_X, tX = feature_extraction(train_X, train_X)
				# if len(np.unique(train_Y)) > 1:
				# 	pred_train_Y = clf.predict(tX)
				# else:
				# 	pred_train_Y = [train_Y[0]] * len(train_Y)
				# if len(pred_train_Y) != len(train_Y):
				# 	print len(train), new_train_X.shape, len(pred_train_Y), len(train_Y)
				# tr, tw = predict_eval(train_Y, pred_train_Y)
				# train_R += tr
				# train_W += tw

				# predictability.append((user.id, len(train), r, w))

				dow_r, dow_w, hod_r, hod_w = predict_eval_by_time(test_X, test_Y, pred_Y)
				dowR = dowR + dow_r
				dowW = dowW + dow_w
				# hodR = hodR + hod_r
				# hodW = hodW + hod_w

				# weekR[(t-11688)/7] += r
				# weekW[(t-11688)/7] += w

		if total_R + total_W > 0:
			accuracy = total_R*1.0/(total_R+total_W)
		else:
			accuracy = -9999
		# if train_R + train_W > 0:
		# 	train_acc = train_R*1.0/(train_R+train_W)
		# else:
		# 	train_acc = -9999
		print (t-start)/7, (total_R+total_W), accuracy
	# bar_plot(hodR, hodW)
	# bar_plot(dowR, dowW)
	# bar_plot(weekR, weekW)
	for i in xrange(len(dowR)):
		print i, dowR[i], dowW[i]


def predict_by_user():
	users = ur.readPanelData("/Users/zhanzhao/Dropbox (MIT)/TfL/Data/sampleData.csv")
	start = 11688
	end = 12419
	user_acc = []
	user_freq = []
	# dowR = np.zeros(7)
	# dowW = np.zeros(7)
	# hodR = np.zeros(24)
	# hodW = np.zeros(24)
	# weekR = np.zeros(105)
	# weekW = np.zeros(105)
	days = range(start, end, 7)
	total_R = 0
	total_W = 0
	# train_R = 0
	# train_W = 0
	for i, user in enumerate(users):
		user_R = 0
		user_W = 0
		sum_train = 0
		count_train = 0
		for t in days:
			# day1 = max(start, (t-7*training_period))
			# day2 = t
			# day3 = t + 7
			train, test = splitDataset(user.tripList, t)
			if len(train) > 0 and len(test) > 0:
				train_X, train_Y = data_prep(train)
				test_X, test_Y = data_prep(test)
				if len(np.unique(train_Y)) > 1:
					new_train_X, new_test_X = feature_extraction(train_X, test_X)
					# clf = nb.MultinomialNB()
					clf = lm.LogisticRegression()
					# clf = tree.DecisionTreeClassifier()
					# clf = svm.LinearSVC()
					clf.fit(new_train_X, train_Y)
					pred_Y = clf.predict(new_test_X)
				else:
					pred_Y = [train_Y[0]] * len(test_Y)
				r, w = predict_eval(test_Y, pred_Y)
				user_R += r
				user_W += w

				count_train += 1
				sum_train += len(train_Y)

				# new_train_X, tX = feature_extraction(train_X, train_X)
				# if len(np.unique(train_Y)) > 1:
				# 	pred_train_Y = clf.predict(tX)
				# else:
				# 	pred_train_Y = [train_Y[0]] * len(train_Y)
				# if len(pred_train_Y) != len(train_Y):
				# 	print len(train), new_train_X.shape, len(pred_train_Y), len(train_Y)
				# tr, tw = predict_eval(train_Y, pred_train_Y)
				# train_R += tr
				# train_W += tw

				# predictability.append((user.id, len(train), r, w))

				# dow_r, dow_w, hod_r, hod_w = predict_eval_by_time(test_X, test_Y, pred_Y)
				# dowR = dowR + dow_r
				# dowW = dowW + dow_w
				# hodR = hodR + hod_r
				# hodW = hodW + hod_w

				# weekR[(t-11688)/7] += r
				# weekW[(t-11688)/7] += w
		total_R += user_R
		total_W += user_W
		if user_R + user_W > 0:
			accuracy = user_R * 1.0 / (user_R + user_W)
			user_acc.append(accuracy)
			avg_train_size = sum_train * 1.0/count_train
			user_freq.append(avg_train_size)
		else:
			accuracy = -9999
		# if train_R + train_W > 0:
		# 	train_acc = train_R*1.0/(train_R+train_W)
		# else:
		# 	train_acc = -9999
		print i, (user_R+user_W), accuracy
	# bar_plot(hodR, hodW)
	# bar_plot(dowR, dowW)
	# bar_plot(weekR, weekW)
	# dist_plot(user_acc)
	scatter_plot(user_freq, user_acc)
	print np.corrcoef(user_freq, user_acc)
	print total_R * 1.0 / (total_R + total_W)


# predict_over_time()
predict_by_user()
