import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class trained_model:

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        for weight in self.has_weight:
            weight.insert_in_annetto()
            rdfWrapper.new_has_weights(self.name,weight.name)

    def add_weight(self,weight):
        self.has_weight.append(weight)


    def __init__(self,name):
        #self.node=node
        self.name=name
        self.has_weight=[]
        self.type="TrainedModel"
