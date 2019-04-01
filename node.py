class Node:
	def __init__(self, layer, index):
		self.layer = layer
		self.index = index
		self.value = 0
		self.weights = {}