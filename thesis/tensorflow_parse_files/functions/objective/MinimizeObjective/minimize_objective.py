from functions.objective.objective import objective
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class minimize_objective(objective):

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        self.cost_function.insert_in_annetto()
        rdfWrapper.new_has_cost(self.name, self.cost_function.name)


    def __init__(self,name,cost_function):
        super(minimize_objective, self).__init__(name)
        self.type="MinObjectiveFunction"
        self.cost_function=cost_function
