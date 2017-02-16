import numpy as np
from parserNLP.Rule import Rule

class Tree(list):
	def __init__(self,label,children):
		list.__init__(self,children)
		self.label=label
	def addChild(self,child):
		self.append(child)
	def __repr__(self):
		if len(self)==0:
			return self.label
		s='('+self.label+' '
		for child in self:
			s+=str(child)
		return s+') '
	def to_cyk(self):
		n = self.get_nterminals()
		cyk_matrix = np.zeros((n,n),dtype=object)
		self.__make_nodes(self,cyk_matrix,0)
		return cyk_matrix
	def __make_nodes(self,t,cyk_matrix, displacement):
		if len(t)==1:
			cyk_matrix[0,displacement] = t.label
		else:    
			cyk_matrix[t.get_nterminals()-1,displacement] = Rule(t.label,t[0].label+' '+t[1].label,0)
			self.__make_nodes(t[0],cyk_matrix,displacement)
			self.__make_nodes(t[1],cyk_matrix,displacement + t[0].get_nterminals())
		return cyk_matrix

	def get_terminals(self,t):
		terminals = []
		if len(t)==1:
			return t[0].label
		terminals += self.get_terminals(t[0])
		terminals += self.get_terminals(t[1])
		return terminals

	def get_nterminals(self):
		return len(self.get_terminals(self))
	@staticmethod
	def from_penn(string):

		#print string
		tree = None
		string = string.strip()
		if not '(' in string:
			#It is a terminal node
			tree = Tree(string,[])
		elif string[0] == '(' and string[len(string)-1] == ')':
			#It is a tree
			content = string[1:len(string)-1].strip()

			if not '(' in content:

				#It is either a terminal node or a preterminal node with a single terminal node

				if not ' ' in content:

					tree = Tree(content.strip(),[])
				else:
					firstBlank = content.index(' ')

					tree = Tree(content[0:firstBlank].strip(),[])
					tree.addChild(Tree(content[firstBlank+1:].strip(),[]))
			else:
				#It is a tree
				firstPar = content.index('(')
				tree = Tree(content[0:firstPar].strip(),[])
				content = content[firstPar:].strip()

				while len(content) > 0:

					openPars = 1;
					index = 1;
					while openPars > 0:

						if content[index] == ')':
							openPars-=1
						elif content[index] == '(':
							openPars+=1
						index+=1
					child=Tree.from_penn(content[0:index].strip())
					tree.addChild(child)
					content = content[index:].strip()
		return tree

if __name__ == '__main__':
	tr='(S (D a) (E b))'
	t = Tree.from_penn(tr)
	print t
	print t.get_terminals(t)
	print t.get_nterminals()
	print t.to_cyk()