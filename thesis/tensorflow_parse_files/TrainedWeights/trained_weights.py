import virtuosoWrapper.virtuosoWrapper as rdfWrapper
class trained_weights:

    def insert_in_annetto(self):
        #print("Annetto::trained_weights-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name,self.type)
        rdfWrapper.new_trained_in_layer(self.name,self.trained_in_layer)
        rdfWrapper.new_trained_out_layer(self.name,self.trained_out_layer)

    def __init__(self,name,tr_in,tr_out):
        self.name=name
        self.type="TrainedWeights"
        self.trained_in_layer = tr_in
        self.trained_out_layer = tr_out
