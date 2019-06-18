from functions.function import function
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class objective(function):

    def insert_in_annetto(self):
        self.cost_function.insert_in_annetto()
        rdfWrapper.new_has_cost(self.name, self.cost_function.name)

    def __init__(self,name):
        super(objective, self).__init__(name,None)
