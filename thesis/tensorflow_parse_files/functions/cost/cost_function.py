import nodes.handler
from functions.RegularizerFunction.l2regularization import l2regularization
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from functions.function import function

class cost_function(function):

    def insert_in_annetto(self):
        #print("Annetto::cost_function-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        self.loss.insert_in_annetto()
        rdfWrapper.new_has_loss(self.name,self.loss.name)
        if self.regularizer!="":
            self.regularizer.insert_in_annetto()
            rdfWrapper.new_has_regularizer(self.name, self.regularizer.name)

    def find_reg(self):
        nm=nodes.handler.entitiesHandler.node_map
        for node_name in nm.keys():
            if nm[node_name].get_op() in nodes.handler.entitiesHandler.regularizers:
                self.regularizer=l2regularization(nm[node_name])


    def __init__(self,name,loss):
        super(cost_function, self).__init__(name,None)
        self.type="CostFunction"
        self.regularizer = ""
        self.loss = loss
        self.find_reg()