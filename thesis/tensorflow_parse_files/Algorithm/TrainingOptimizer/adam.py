from Algorithm.TrainingOptimizer.training_optimizer import training_optimizer
import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler

class adam(training_optimizer):

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_learning_rate(self.name, self.learning_rate)

    def find_learning_rate(self):
        super.find_learning_rate(adam, self)

    def __init__(self,node,name):
        super(adam, self).__init__(node,name)
        name=self.name.split("/")[0]
        self.type="Adam"
        self.find_learning_rate()
