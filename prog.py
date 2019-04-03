import random
import sys
import math
from node import *
from constant import *


def getSmallRandom():
	# Random between -MAX_INIT and MAX_INIT
	return round(random.uniform(0, MAX_INIT) * random.choice([-1, 1]), ROUND_TO)

	# Random between 0 and MAX_INIT
	# return round(random.uniform(0, MAX_INIT), ROUND_TO)


def normalizeDiscrete(x):
	return (x - XMIN) / (XMAX - XMIN)


def hiddenActivationFunction(sigma):
	# Threshold Function
	# if sigma > 0:
	# 	return 1
	# else:
	# 	return 0

	# Bipolar Sigmoid
	# return round((2 / (1 + (math.e ** (sigma * -1))) - 1, ROUND_TO)

	# Binary Sigmoid
	return 1 / (1 + (math.e ** (-1 * sigma)))

	# Rectified Linear Unit (ReLU)
	# return max(0, sigma)

	# Leaky ReLU
	# if sigma >= 0:
	# 	return sigma
	# else:
	# 	return (1/20) * sigma


def hiddenErrorFunction(sigma):
	# Sigmoid Derivative
	return hiddenActivationFunction(sigma) * (1 - hiddenActivationFunction(sigma))

	# Leaky ReLU Derivative
	# sigma >= 0:
	# 	return 1
	# else:
	# 	return (1/20)


def outputActivationFunction():
	# Threshold
	y = net[numLayers][0].value
	if y == 1:
		return 9
	else:
		return math.floor(y * 10)

	# Softmax
	# outputs = []
	# for node in net[numLayers].values():
	# 	outputs.append(node.value)
	# inputMax = max(outputs)
	# exps = [math.e ** (i - inputMax) for i in outputs]
	# sum_of_exps = sum(exps)
	# softmax = [i/sum_of_exps for i in exps]
	# return softmax.index(max(softmax))


def outputErrorFunction(output, target):
	# Sigmoid
	# return output * (target - output) * (1 - output)

	# Softmax
	# outputs = []
	# for node in net[numLayers].values():
	# 	outputs.append(node.value)
	# inputMax = max(outputs)
	# exps = [math.e ** (i - inputMax) for i in outputs]
	# sum_of_exps = sum(exps)
	# softmax = [i/sum_of_exps for i in exps]
	# thisSum = sum(softmax)
	# return softmax[output] - softmax[target]
	# return -1 * math.log(softmax[softmaxIndex]/thisSum)

	# Sum of Squares
	result = 0
	for i in net[numLayers]:
		if i == target:
			result += (net[numLayers][i] - 1) ** 2
		else:
			result += (net[numLayers][i] - 0) ** 2
	return result

	# Cross Entropy
	# result = 0
	# for i in net[numLayers]:
	# 	result += net[numLayers][i].value * math.log(net[numLayers][i].value)
	# return result


def backpropagate(target):
	output = outputActivationFunction()
	outputError = outputErrorFunction(net[numLayers][0].value, (target/10) + .05)

	### Calculate Errors
	errors = {}
	for i in range(numLayers + 1):
		errors[i] = {}

	# Inner hidden layers
	for i in range(1, numLayers):
		for j in range(numNodes + 1):
			# Layer before output layers
			if i == numLayers - 1:
				for k in range(NUMOUTPUTNODES):
					errors[i][j] = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[k] * outputError)
			else:
				for k in range(numNodes):
					errors[i][j] = net[i][j].value * (1 - net[i][j].value) * (net[i][j].weights[k] * outputError)

	# Output layer
	for i in range(NUMOUTPUTNODES):
		errors[numLayers][i] = outputError

	### Learn (adjust weights)
	# Initial layer --> layer 1
	for i in range(NUMINITNODES + 1):
		for j in range(numNodes):
			net[0][i].weights[j] += LEARNING_FACTOR * errors[0 + 1][j] * net[0][i].value

	# Inner hidden layers
	for i in range(1, numLayers):
		for j in range(numNodes + 1):
			# Layer before output layer
			if i == numLayers - 1:
				for k in range(NUMOUTPUTNODES):
					net[i][j].weights[k] += LEARNING_FACTOR * errors[i + 1][k] * net[i][j].value
			else:
				for k in range(numNodes):
					net[i][j].weights[k] += LEARNING_FACTOR * errors[i + 1][k] * net[i][j].value


def feedforward(target):
	# Initial layer --> layer 1
	for i in range(numNodes):
		sop = 0
		for j in range(NUMINITNODES):
			sop += net[0][j].value * net[0][j].weights[i]
		sop += net[0][NUMINITNODES].value * net[0][NUMINITNODES].weights[i]
		net[1][i].value = hiddenActivationFunction(sop)

	# Inner hidden layers
	for i in range(1, numLayers - 1):
		for j in range(numNodes):
			sop = 0
			for k in range(numNodes):
				sop += net[i][k].value * net[i][k].weights[j]
			sop += net[i][numNodes].value * net[i][numNodes].weights[j]
			net[i+1][j].value = hiddenActivationFunction(sop)

	# Layer before output layer
	for j in range(NUMOUTPUTNODES):
		i = numLayers - 1
		sop = 0
		for k in range(numNodes):
			sop += net[i][k].value * net[i][k].weights[j]
		sop += net[i][numNodes].value * net[i][numNodes].weights[j]
		net[numLayers][j].value = hiddenActivationFunction(sop)


def printNet():
	print('Neural Net:')
	print('Output: %i\n' % (outputActivationFunction()))
	print('Output Layer')
	for i in range(NUMOUTPUTNODES):
		print('\t%i: %.3f' %(i, net[numLayers][i].value))
	print()
	for i in reversed(range(1, numLayers)):
		print('Layer',i)
		for j in range(numNodes + 1):
			if j == numNodes:
				print('\tB: %.3f' % (net[i][j].value), end=' ')
			else:
				print('\t%i: %.3f' % (j, net[i][j].value), end=' ')
			print('[',end=' ')
			for k, v in net[i][j].weights.items():
				print('%i: %.3f' % (k, v), end=' ')
			print(']')
		print()
	print('Initial Layer')
	for i in range(NUMINITNODES + 1):
		if i == NUMINITNODES:
			print('\tB: %.3f' % (net[0][i].value), end=' ')
		else:
			print('\t%i: %.3f' % (i, net[0][i].value), end=' ')
		print('[',end=' ')
		for k, v in net[0][i].weights.items():
			print('%i: %.3f' % (k, net[0][i].weights[k]), end=' ')
		print(']')
	print()


if __name__ == '__main__':
	if (len(sys.argv)) != 3:
		print('Usage error: prog.py [train filename] [test filename]')
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
		net[i][numNodes].value = BIAS_VALUE

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
	net[0][64].value = BIAS_VALUE
	for i in range(numNodes):
		net[0][64].weights[i] = getSmallRandom()

	isFirst = True
	# Training (data format in constants.py)
	with open(sys.argv[1], 'r') as f:
		for epoch in range(EPOCHS):
			for line in f.readlines():
				tokens = line.strip().split()
				for i in range(64):
					net[0][i].value = normalizeDiscrete(int(tokens[i]))
				target = int(tokens[64])
				if isFirst:
					isFirst = False
					print('\nInitial', end=' ')
					printNet()
				feedforward(target)
				backpropagate(target)
				print('Target:',target)
				printNet()

	print('Training complete. Testing...')

	# Testing
	correct = 0
	total = 0
	with open(sys.argv[2], 'r') as f:
		for line in f.readlines():
			tokens = line.strip().split()
			for i in range(64):
				net[0][i].value = normalizeDiscrete(int(tokens[i]))
			target = int(tokens[64])
			feedforward(target)
			if outputActivationFunction() == target:
				correct += 1
			total += 1

	print('Got',correct,'/',total,'correct, or',10*round(correct/total, 3),'%')