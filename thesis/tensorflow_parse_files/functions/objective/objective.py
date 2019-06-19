from functions.function import function
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class objective(function):

    def __init__(self,name):
        super(objective, self).__init__(name,None)
