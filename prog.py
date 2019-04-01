import random
import sys
import math
from node import *
from constant import *


def getSmallRandom():
	return round(random.uniform(0, MAX_INIT) * random.choice([-1, 1]), 3)


def normalizeDiscrete(x):
	return (x - XMIN) / (XMAX - XMIN)


def activationFunction(sigma):
	# if sigma > 0:
	# 	return 1
	# else:
	# 	return 0

	return round((2 / (1 + math.pow(math.e, sigma * -1))) - 1, 3)


def outputActivation(y):
	if y <= .1:
		return 0
	elif y <= .2:
		return 1
	elif y <= .3:
		return 2
	elif y <= .4:
		return 3
	elif y <= .5:
		return 4
	elif y <= .6:
		return 5
	elif y <= .7:
		return 6
	elif y <= .8:
		return 7
	elif y <= .9:
		return 8
	else:
		return 9

def backpropagate(numLayers, numNodes, target):
	output = outputActivation(net[numLayers][0].value)
	outputError = output * (1 - output) * (target - output)

	for i in range(NUMINITNODES):
		for j in range(numNodes):
			error = net[0][i].value * (1 - net[0][i].value) * (net[0][i].weights[j] * outputError)
			net[0][i].weights[j] = round(net[0][i].weights[j] + (LEARNING_FACTOR * outputError * net[0][i].value), 3)

	for i in range(1, numLayers):
		for j in range(numNodes):
			if i == numLayers - 1:
				error = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[0] * outputError)
				net[i][j].weights[0] = round(net[i][j].weights[0] + (LEARNING_FACTOR * outputError * net[i][j].value), 3)
			else:
				for k in range(numNodes):
					error = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[k] * outputError)
					net[i][j].weights[k] = round(net[i][j].weights[k] + (LEARNING_FACTOR * outputError * net[i][j].value), 3)


def train(numLayers, numNodes, target):
	# Initial layer --> layer 1
	for i in range(numNodes):
		sop = 0
		for j in range(NUMINITNODES):
			sop += (net[0][j].value * net[0][j].weights[i])
		sop += (net[0][NUMINITNODES].value * net[0][NUMINITNODES].weights[i])
		net[1][i].value = activationFunction(sop)

	# Rest of layers
	for i in range(1, numLayers):
		# Layer before output node
		if (i == numLayers - 1):
			sop = 0
			for j in range(numNodes):
				sop += net[i][j].value * net[i][j].weights[0]
				sop += net[i][numNodes].value * net[i][numNodes].weights[0]
			net[numLayers][0].value = sop
		else:
			for j in range(numNodes):
				sop = 0
				for k in range(numNodes):
					sop += net[i][k].value * net[i][k].weights[j]
				sop += net[i][numNodes].value * net[i][numNodes].weights[j]
				net[i+1][j].value = activationFunction(sop)

	backpropagate(numLayers, numNodes, target)

def printNet(numLayers, numNodes):
	print('Neural Net:')
	print('Output: %.3f (%i)' % (net[numLayers][0].value, outputActivation(net[numLayers][0].value)))
	for i in reversed(range(1, numLayers)):
		print('Layer',i)
		for j in range(numNodes + 1):
			print('\t%.3f' % (net[i][j].value), end=' ')
			print(net[i][j].weights)
		print()
	print('Initial Layer')
	for i in range(NUMINITNODES + 1):
		print('\t', net[0][i]. value, end=' ')
		print(net[0][i].weights)
	print()


if __name__ == '__main__':
	if (len(sys.argv)) != 2:
		print('Usage error: prog.py [filename]')
		sys.exit()
	try:
		numLayers = int(input('Number of hidden layers (2): '))
	except:
		numLayers = 2
	try:
		numNodes = int(input('Number of nodes in each layer (3): '))
	except:
		numNodes = 3

	# Neural Net initialization
	numLayers += 1
	net = {}
	random.seed(RANDOM_SEED)
	for i in range(1, numLayers):
		net[i] = {}
		for j in range(numNodes):
			net[i][j] = Node(i, j)
		net[i][numNodes] = Node(i, numNodes) # bias node
		net[i][numNodes].value = 1
	net[numLayers] = {}
	net[numLayers][0] = Node(numLayers, 0) # output node

	# Add random weights
	for i in range(1, numLayers):
		# numNodes + 1 for bias node
		for j in range(numNodes + 1):
			# Last hidden layer points to single output node
			if i == numLayers - 1:
				net[i][j].weights[0] = getSmallRandom()
			else:
				# Randomize weight for connection to each node
				# in next layer (not including bias node)
				for k in range(numNodes):
					net[i][j].weights[k] = getSmallRandom()

	# Initial input layer specific to digits data
	net[0] = {}
	for i in range(64):
		net[0][i] = Node(0, i)
		for j in range(numNodes):
			net[0][i].weights[j] = getSmallRandom()
	net[0][64] = Node(0, 64)
	net[0][64].value = 1
	for i in range(numNodes):
		net[0][64].weights[i] = getSmallRandom()

	print('\nInitial', end=' ')
	printNet(numLayers, numNodes)

	# Training (data format in constants.py)
	with open(sys.argv[1], 'r') as f:
		for line in f.readlines():
			tokens = line.strip().split()
			for i in range(64):
				net[0][i].value = int(tokens[i])
			target = int(tokens[64])
			train(numLayers, numNodes, target)
			print('Target:',target)
			printNet(numLayers, numNodes)