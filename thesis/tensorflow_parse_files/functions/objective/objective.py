from functions.function import function
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
class objective(function):


    def insert_in_annetto(self):
        #print("Annetto::objective-", self.name)
        #nodes.handler.entitiesHandler.new_activation_subclass(self.name, self.subclass)
        self.cost_function.insert_in_annetto()
        rdfWrapper.new_has_cost(self.name, self.cost_function.name)

    def __init__(self,name):
        super(objective, self).__init__(name,None)
        #self.is_in=self.check_if_in(name)
