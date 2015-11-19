import csv
import pandas as pd


class User:
	def __init__(self, ID, cardType=None, tripList=None):
		self.id = ID
		self.cardType = cardType
		self.tripList = tripList

	def addTrip(self, trip):
		self.tripList.append(trip)

	def getTripList(self):
		return self.tripList

	def getOriginList(self):
		return [item[2] for item in self.getTripList()]

	def getDestinationList(self):
		return [item[4] for item in self.getTripList()]

	def getODList(self):
		return [(item[2], item[4]) for item in self.getTripList()]

	def writeUserTrips(self, file):
		trips = self.tripList
		myWriter = csv.writer(open(file, 'wb'), delimiter=',')
		myWriter.writerow(['daykey', 'time1', 'station1', 'time2', 'station2'])
		for row in trips:
			myWriter.writerow(row)

	def getUserNetwork(self):
		tripDF = pd.DataFrame(self.tripList)
		tripDF.column = ['daykey', 'time1', 'station1', 'time2', 'station2']
		links = tripDF.groupby([2, 4]).size()
		return links

	def writeUserNetwork(self, file):
		X = []
		links = self.getUserNetwork()
		for (o, d), count in links.iteritems():
			X.append((o, d, count))

		def getKey(item):
			return item[2]

		X.sort(key=getKey)
		myWriter = csv.writer(open(file, 'wb'), delimiter=',')
		myWriter.writerow(['origin', 'destination', 'count'])
		for row in X:
			myWriter.writerow(row)

	def getOriginRank(self):
		X = []
		tripDF = pd.DataFrame(self.tripList)
		tripDF.column = ['daykey', 'time1', 'station1', 'time2', 'station2']
		origins = tripDF.groupby(2).size()
		for o, count in origins.iteritems():
			X.append((o, count))

		def getKey(item):
			return item[1]

		X.sort(key=getKey, reverse=True)
		return X

	def getDestinationRank(self):
		X = []
		tripDF = pd.DataFrame(self.tripList)
		tripDF.column = ['daykey', 'time1', 'station1', 'time2', 'station2']
		destinations = tripDF.groupby(4).size()
		for d, count in destinations.iteritems():
			X.append((d, count))

		def getKey(item):
			return item[1]

		X.sort(key=getKey, reverse=True)
		return X

	def getODRank(self):
		X = []
		links = self.getUserNetwork()
		for (o, d), count in links.iteritems():
			X.append(((o, d), count))

		def getKey(item):
			return item[1]

		X.sort(key=getKey, reverse=True)
		return X

	def getActiveDays(self):
		trips = self.getTripList()
		days = []
		prev_day = -1
		for t in trips:
			if int(t[0]) > prev_day:
				days.append(t[0])
				prev_day = int(t[0])
		return len(days)


class panelDataReader:
	def __init__(self, file):
		self.reader = csv.reader(open(file), delimiter=";")
		self.header = self.reader.next()
		self.lastRecord = None

	def nextRecord(self):
		try:
			line = next(self.reader)
		except StopIteration:
			line = None
		self.lastRecord = line
		return line

	def nextUserRecords(self):
		records = []
		if self.lastRecord is None:
			if self.nextRecord() is None:
				return None

		firstRecord = self.lastRecord
		records.append(firstRecord)
		while True:
			prevID = self.lastRecord[0]
			nextRecord = self.nextRecord()
			if nextRecord is not None and prevID == nextRecord[0]:
				records.append(nextRecord)
			else:
				break

		if len(records) > 0:
			return records
		else:
			return None


def readData(file):
	df = pd.read_csv(file, sep=";")
	return df


def readPanelData(file):
	print 'Importing users...'
	X = []
	counter = 0
	panelReader = panelDataReader(file)
	records = panelReader.nextUserRecords()
	while records is not None:
		userID = records[0][0]
		userCard = records[0][1]
		userRecords = []
		for i in xrange(len(records)-1):
			tap1 = records[i][3:]
			tap2 = records[i+1][3:]
			daykey = int(tap1[0])
			if tap1[1] == '61' and tap2[1] == '62' and \
				daykey >= 11688 and daykey < 12419:
				trip = (daykey, int(tap1[2]), tap1[4], int(tap2[2]), tap2[5])
				userRecords.append(trip)
		newUser = User(userID, cardType=userCard, tripList=userRecords)
		X.append(newUser)
		counter += 1
		if counter % 100 == 0:
			print counter
		records = panelReader.nextUserRecords()
	return X
