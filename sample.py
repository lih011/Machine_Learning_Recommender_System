
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def readCSV(path):
    """Function to read data.
    Input: path:(path-like string) directory path of data
    Return: None
    """
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
bookRatings = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append((user,book,r))
    userRatings[user].append(r)
    bookRatings[book].append(r)

globalAverage = sum([r[2] for r in allRatings]) / len(allRatings)

userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

bookAverage = {}
for b in bookRatings:
    bookAverage[b] = sum(bookRatings[b]) / len(bookRatings[b])


### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

def Jaccard(s1, s2):
    """Function to compute Jaccard similarity.
    Inputs: s1, s2: (sets)
    Return: Jaccard Similarity (float)
    """
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


##### Task: Predict Read #####

def JaccardBasedPredict(usersPerItem,itemsPerUser,user,item,jThreshold):
    """Function to predict read based on the Jaccard similarity threshold (using user similarity).
    Inputs: (1) usersPerItem: (dict) key - item ID, value - list of user IDs
            (2) itemsPerUser: (dict) key - user ID, value - list of item IDs
            (3) user: (string) user ID
            (4) book: (string) book ID
            (5) jThreshold: (float) Jaccard similarity threshold
    Return: Read: (boolean) 1 - read, 0 - not read
    """
    J_list = []
    users = usersPerItem[item]
    items = itemsPerUser[user]
    if len(items) == 0:
            J_list.append(0)
    else:
        for item2 in items:
            users2 = usersPerItem[item2]
            J_list.append(Jaccard(set(users),set(users2)))
    maxJ = max(J_list)
    if maxJ > jThreshold:
        return True
    else:
        return False

def JaccardBasedPredict2(usersPerItem,itemsPerUser,user,item,jThreshold):
    """Function to predict read based on the Jaccard similarity threshold (using item similarity).
    Inputs: (1) usersPerItem: (dict) key - item ID, value - list of user IDs
            (2) itemsPerUser: (dict) key - user ID, value - list of item IDs
            (3) user: (string) user ID
            (4) book: (string) book ID
            (5) jThreshold: (float) Jaccard similarity threshold
    Return: Read: (boolean) 1 - read, 0 - not read
    """
    J_list = []
    users = usersPerItem[item]
    items = itemsPerUser[user]
    if len(users) == 0:
            J_list.append(0)
    else:
        for user2 in users:
            items2 = itemsPerUser[user2]
            J_list.append(Jaccard(set(items),set(items2)))
    maxJ = max(J_list)
    if maxJ > jThreshold:
        return True
    else:
        return False

def PopularityBasedPredict(usersPerItem,item,popThreshold):
    """Function to predict read based on the popularity threshold.
    Inputs: (1) usersPerItem: (dict) key - item ID, value - list of user IDs
            (2) item: (string) item ID
            (3) popThreshold: (int) popoluarity threshold
    Return: Read: (boolean) 1 - read, 0 - not read
    """
    if len(usersPerItem[item]) >= popThreshold:
        return  True
    else:
        return False

def predict_read(usersPerItem,itemsPerUser,user,item,popThreshold,jThreshold):
    """Function to predict read based on the popularity threshold and Jaccard similarity  threshold.
    Inputs: (1) usersPerItem: (dict) key - item ID, value - list of user IDs
            (2) itemsPerUser: (dict) key - user ID, value - list of item IDs
            (3) user: (string) user ID
            (4) book: (string) book ID
            (5) popThreshold: (int) popularity threshold
            (6) jThreshold: (float) Jaccard similarity threshold
    Return: Read: (boolean) 1 - read, 0 - not read
    """
    Jacc = JaccardBasedPredict(usersPerItem,itemsPerUser,user,item,jThreshold)
    Pop = PopularityBasedPredict(usersPerItem,item,popThreshold)
    return (Jacc or Pop) 

def evaluate_accuracy(labels,preds):
    """Function to evaluate the predictions with accuracy.
    Inputs: (1) labels: (list) observations
            (2) preds: (list) predictions
    Return: Accuracy: (float) accuracy of predicts
    """
    assert len(labels) == len(preds)
    return np.equal(labels,preds).sum()/len(labels)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
booksPerUserAll = defaultdict(list)
usersPerBookAll = defaultdict(list)
booksPerUserTrain = defaultdict(list)
usersPerBookTrain = defaultdict(list)

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    booksPerUserTrain[u].append(b)
    usersPerBookTrain[b].append(u)

for u,b,r in allRatings:
    booksPerUserAll[u].append(b)
    usersPerBookAll[b].append(u)


readValid = [(r[0],r[1],True) for r in ratingsValid]  # positives


def construct_valid_sample(readValid,itemCount,itemsPerUser):
    """Function to construct a balanced validation sample with half postives and half negatives.
    Inputs: (1) readValid: (dict) original valid sample
            (2) itemCount: (dict) key - item ID, value - # of user reviews
            (3) itemsPerUser: (dict) key - user ID, value - list of item IDs
    Return: readValid2: (dict) expanded validation sample
    """
    readValid2 = readValid.copy()
    # add negatives:
    for i in range(len(readValid2)):
        u = readValid[i][0]
        b = readValid[i][1]
        while b in itemsPerUser[u]:
            b = random.choice(list(itemCount.keys()))
        readValid2.append([u,b,False])
    return readValid2


readValid2 = construct_valid_sample(readValid,bookCount,booksPerUserAll)


labelsValid = [r[2] for r in readValid2]


def find_optimal_popularity_threshold(pLow,pHigh,usersPerItem,readValid):
    acc = []
    for p in range(pLow,pHigh+1,1):
        acc.append(evaluate_accuracy(labelsValid,[PopularityBasedPredict(usersPerItem,r[1], p) for r in readValid]))
    return acc,max(acc),acc.index(max(acc))+1


pLow = min(list(bookCount.values()))
pHigh = max(list(bookCount.values()))


acc_pop,bestPopPredAccuracy,optPopThreshold = find_optimal_popularity_threshold(pLow,pHigh,usersPerBookTrain,readValid2)


print(bestPopPredAccuracy)

print(optPopThreshold)


plt.plot(np.arange(pLow,pHigh+1,1),acc_pop)


def find_optimal_J_threshold(jRange,usersPerItem,itemsPerUser,readValid,popThreshold):
    max_acc = -np.inf
    for ix in range(len(jRange)):
        j = jRange[ix]
        pred = [predict_read(usersPerItem,itemsPerUser,r[0],r[1],popThreshold,j1) for r in readValid]
        acc = evaluate_accuracy(labelsValid,pred)
        if acc > max_acc:
            max_acc = acc
            optJ = j
    return max_acc,optJ

j_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]

max_acc,optJ = find_optimal_J_threshold(j_list,usersPerBookTrain,booksPerUserTrain,readValid2,30)


print(max_acc)

print(optJ)


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    read = int(predict_read(usersPerBookAll,booksPerUserAll,u,b,30,0.05))
    line = str(u) + ',' + str(b) + ',' + str(read) + '\n'
    predictions.write(line)
predictions.close()


##### Task: Predict Rating (CSE258 Only) #####

N = len(allRatings)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())

ratingMean = sum([r[2] for r in ratingsTrain])/len(ratingsTrain)

alpha0 = ratingMean # initial guess for constant

betaU0 = {u:0 for u in ratingsPerUser} # initial guess for user biases

betaI0 = {i:0 for i in ratingsPerItem} # initial guess for item biases


def SimpleLatentFactorPredict(alpha,betaU,betaI,user,item):
    """Function to predict ratings based on the simple latent factor model.
    """
    return alpha + betaU[user] + betaI[item]


def MSE(predictions, labels):
    """Function to compute MSE.
    """
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


def update_params(alpha0,betaU0,betaI0,lamb):
    """Function to dynamically update model parameters based on the resulting MSE.
    """
    # update alpha:
    alpha =  alpha0
    diffAlpha = 0
    for u,i,r in ratingsTrain:
        diffAlpha += r - betaU0[u] - betaI0[i] - alpha0
    alpha = alpha0 + diffAlpha/len(ratingsTrain) 
    
    # update Beta U:
    betaU = betaU0.copy()
    diffBetaU = {u:0 for u in betaU}
    for u in ratingsPerUser:
        for i,r in ratingsPerUser[u]:
            diffBetaU[u] += r - alpha - betaI0[i] - betaU0[u]
        betaU[u] = (len(ratingsPerUser[u])*betaU0[u] + diffBetaU[u])/(lamb + len(ratingsPerUser[u]))
    
    # update Beta I:
    betaI = betaI0.copy()
    diffBetaI = {i:0 for i in betaI}
    for i in ratingsPerItem:
        for u,r in ratingsPerItem[i]:
            diffBetaI[i] += r - alpha - betaU[u] - betaI0[i]
        betaI[i] = (len(ratingsPerItem[i])*betaI0[i] + diffBetaI[i])/(lamb + len(ratingsPerItem[i]))
    
    # make predictions and compute related MSE
    pred = [SimpleLatentFactorPredict(alpha,betaU,betaI,u,i) for u,i,_ in ratingsTrain]
    labels = [r[2] for r in ratingsTrain]
    mse = MSE(pred,labels)
    
    # update regularizer value:
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for i in betaI:
        regularizer += betaI[i]**2
    
    return alpha, betaU, betaI, mse, mse + lamb * regularizer


def search_optimal(apha0,betaU0,betaI0,lamb,maxIterations,tol):
    """Function to search optimal parameters for predicting ratings.
    """
    alpha, betaU, betaI, mse, cost = update_params(alpha0,betaU0,betaI0,lamb)
    print("In Iteration 1, the MSE is {}, and Objective Cost to Minimize is {}".format(mse,cost))
    alpha, betaU, betaI, mse2, cost2 = update_params(alpha,betaU,betaI,lamb)
    print("In Iteration 2, the MSE is {}, and Objective Cost to Minimize is {}".format(mse2,cost2))
    n =  2
    while abs(cost2 - cost) > tol:
        n += 1 
        if n > maxIterations: break
        mse, cost = mse2, cost2
        alpha, betaU, betaI, mse2, cost2 = update_params(alpha,betaU,betaI,lamb)
        print("In Iteration {}, the MSE is {}, and Objective Cost to Minimize is {}".format(n,mse,cost))
    return alpha, betaU, betaI, mse2, cost2


def tune_lamb(alpha0,betaU0,betaI0,lambRange,maxIterations,tolObj):
    """Function to tune lambda parameter using the validation set.
    """
    ratingValuesValid = [r[2] for r in ratingsValid]
    newAlpha = alpha0
    newBetaU = betaU0.copy()
    newBetaI = betaI0.copy()
    newLamb = lambRange[0]
    mseOpt = np.inf
    costOpt = np.inf
    
    for lamb in lambRange:
        print("In the case Lambda = {}".format(lamb))
        alpha,betaU,betaI,mse,cost = search_optimal(alpha0,betaU0,betaI0,lamb,maxIterations,tolObj)
        pred = []
        for u,i,r in ratingsValid:
            if u in betaU and i in betaI:
                pred.append(SimpleLatentFactorPredict(alpha,betaU,betaI,u,i))
            else:
                if u in userAverage:
                    pred.append(userAverage[u])
                elif i in bookAverage:
                    pred.append(bookAverage[i])
                else:
                    pred.append(globalAverage)
        mse = MSE(pred,ratingValuesValid)
        if mse < mseOpt:
            mseOpt = mse
            costOpt = cost
            newAlpha = alpha
            newBetaU = betaU.copy()
            newBetaI = betaI.copy()
            newLamb = lamb
        print()
    return newAlpha,newBetaU,newBetaI,newLamb,mseOpt,costOpt


alphaOpt,betaUOpt,betaIOpt,lambOpt,mseOpt,costOpt = tune_lamb(alpha0,betaU0,betaI0,range(1,10),200,0.0001)

alphaOpt,betaUOpt,betaIOpt,lambOpt,mseOpt,costOpt = tune_lamb(alpha0,betaU0,betaI0,np.arange(4.5,5.6,0.1),200,0.0001)

print(lambOpt)

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split(',')
    if u in betaUOpt and i in betaIOpt:
        rating = SimpleLatentFactorPredict(alphaOpt,betaUOpt,betaIOpt,u,i)
    else:
        if u in userAverage:
            rating = userAverage[u]
        elif i in bookAverage:
            rating = bookAverage[i]
        else:
            rating = globalAverage
    line = str(u) + ',' + str(i) + ',' + str(rating) + '\n'
    predictions.write(line)

predictions.close()
