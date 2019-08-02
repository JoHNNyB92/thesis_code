from Algorithm.algorithm import algorithm
import nodes.handler

class training_optimizer(algorithm):

    def find_learning_rate(self):
        nm = nodes.handler.entitiesHandler.node_map
        for key in nm.keys():
            #if key.endswith("/learning_rate"):
            if key.endswith(self.name+"/learning_rate"):
                self.learning_rate = nm[key].get_attr()["value"].tensor.float_val[0]
                return
        print("ERROR:Unable to find learning rate for ",self.name,". Setting it to -1.")
        self.learning_rate=-1

    def __init__(self,node,name):
        super(training_optimizer, self).__init__(node,name)