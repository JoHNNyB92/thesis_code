from functions.objective.objective import objective
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class maximize_objective(objective):

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        self.cost_function.insert_in_annetto()
        rdfWrapper.new_has_cost(self.name, self.cost_function.name)
        super(maximize_objective, self).insert_in_annetto()


    def __init__(self,name,cost_function):
        super(maximize_objective, self).__init__(name)
        self.type="MaxObjectiveFunction"
        self.cost_function=cost_function
