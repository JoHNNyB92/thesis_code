from Algorithm.algorithm import algorithm
import nodes.handler

class training_optimizer(algorithm):

    def find_learning_rate(self, name):
        nm = nodes.handler.entitiesHandler.node_map
        for key in nm.keys():
            if key.endswith("/learning_rate"):
                self.learning_rate = nm[key].get_attr()["value"].tensor.float_val[0]

    def __init__(self,node,name):
        super(training_optimizer, self).__init__(node,name)