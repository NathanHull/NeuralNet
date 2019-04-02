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
	# Threshold Function
	# if sigma > 0:
	# 	return 1
	# else:
	# 	return 0

	# Bipolar Sigmoid
	# return round((2 / (1 + (math.e ** (sigma * -1))) - 1, 3)

	# Binary Sigmoid
	return round(1 / (1 + (math.e ** (-1 * sigma))), 3)


def outputActivation():
	# Threshold
	# if y == 1:
	# 	return 9
	# else:
	# 	return math.floor(y * 10)

	# Softmax
	outputs = []
	for node in net[numLayers].values():
		outputs.append(node.value)
	exps = [math.e ** i for i in outputs]
	sum_of_exps = sum(exps)
	softmax = [i/sum_of_exps for i in exps]
	return softmax.index(max(softmax))


def reverseOutput(y):
	return (y / 10) + .05


def backpropagate(target):
	output = net[numLayers][0].value
	target = reverseOutput(target)
	outputError = output * (1 - output) * (target - output)

	for i in range(NUMINITNODES):
		for j in range(numNodes):
			error = net[0][i].value * (1 - net[0][i].value) * (net[0][i].weights[j] * outputError)
			net[0][i].weights[j] += round(LEARNING_FACTOR * error * net[0][i].value, 3)

	for i in range(1, numLayers):
		for j in range(numNodes):
			if i == numLayers - 1:
				error = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[0] * outputError)
				net[i][j].weights[0] += round(LEARNING_FACTOR * error * net[i][j].value, 3)
			else:
				for k in range(numNodes):
					error = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[k] * outputError)
					net[i][j].weights[k] += round(LEARNING_FACTOR * error * net[i][j].value, 3)


def feedforward(target):
	# Initial layer --> layer 1
	for i in range(numNodes):
		sop = 0
		for j in range(NUMINITNODES):
			sop += (normalizeDiscrete(net[0][j].value) * net[0][j].weights[i])
		sop += (net[0][NUMINITNODES].value * net[0][NUMINITNODES].weights[i])
		net[1][i].value = activationFunction(sop)

	# Rest of layers
	for i in range(1, numLayers):
		# Layer before output node
		if (i == numLayers - 1):
			for j in range(numNodes):
				sop = 0
				for k in range(NUMOUTPUTNODES):
					sop += net[i][j].value * net[i][j].weights[k]
					sop += net[i][numNodes].value * net[i][numNodes].weights[k]
				net[numLayers][j].value = sop
		else:
			for j in range(numNodes):
				sop = 0
				for k in range(numNodes):
					sop += net[i][k].value * net[i][k].weights[j]
				sop += net[i][numNodes].value * net[i][numNodes].weights[j]
				net[i+1][j].value = activationFunction(sop)

	backpropagate(target)

def printNet():
	print('Neural Net:')
	print('Output: %i\n' % (outputActivation()))
	print('Output Layer')
	for i in range(NUMOUTPUTNODES):
		print('\t%i: %.3f' %(i, net[numLayers][i].value))
	print()
	for i in reversed(range(1, numLayers)):
		print('Layer',i)
		for j in range(numNodes + 1):
			if j == numNodes + 1:
				print('\tB: %.3f' % (net[i][j].value), end=' ')
			else:
				print('\t%i: %.3f' % (j, net[i][j].value), end=' ')
			print(net[i][j].weights)
		print()
	print('Initial Layer')
	for i in range(NUMINITNODES + 1):
		if i == NUMINITNODES + 1:
			print('\tB: %.3f' % (net[0][i].value), end=' ')
		else:
			print('\t%i: %.3f' % (i, net[0][i].value), end=' ')
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
	for i in range(NUMOUTPUTNODES):
		net[numLayers][i] = Node(numLayers, i) # output nodes

	# Add random weights
	for i in range(1, numLayers):
		# numNodes + 1 for bias node
		for j in range(numNodes + 1):
			# Last hidden layer points to single output node
			if i == numLayers - 1:
				for k in range(NUMOUTPUTNODES):
					net[i][j].weights[k] = getSmallRandom()
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
	printNet()

	# Training (data format in constants.py)
	with open(sys.argv[1], 'r') as f:
		for line in f.readlines():
			tokens = line.strip().split()
			for i in range(64):
				net[0][i].value = int(tokens[i])
			target = int(tokens[64])
			feedforward(target)
			print('Target:',target)
			printNet()