from Algorithm.TrainingOptimizer.training_optimizer import training_optimizer
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler

class rms_prop(training_optimizer):

    def insert_in_annetto(self):
        #print("Annetto::rms_prop-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_learning_rate(self.name, self.learning_rate)

    def find_learning_rate(self, name):
        nm=nodes.handler.entitiesHandler.node_map
        for key in nm.keys():
            if key.endswith("/learning_rate"):
                self.learning_rate = nm[key].get_attr()["value"].tensor.float_val[0]

    def __init__(self,node):
        super(rms_prop, self).__init__(node,node.get_name())
        name=self.name.split("/")[0]
        self.type="RMSProp"
        self.find_learning_rate(name)
