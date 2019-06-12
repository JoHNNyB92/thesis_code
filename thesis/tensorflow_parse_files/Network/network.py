import virtuosoWrapper.virtuosoWrapper as rdfWrapper
class network:
    def insert_in_annetto(self):
        #print("Annetto::Network-",self.name)
        for layer_name in self.layer.keys():
            self.layer[layer_name].insert_in_annetto()
            rdfWrapper.new_network_has_layer(self.name, layer_name)
            #TODO: inputLayer/OutputLayer/Tasktype
        if self.objective!="":
            for objective_name in self.objective.keys():
                self.objective[objective_name].insert_in_annetto()
                rdfWrapper.new_network_has_objective(self.name, objective_name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_network(self.name)
        for elem in self.input_layer:
            elem.insert_in_annetto()
            rdfWrapper.new_input_layer(self.name,elem.name)
        for elem in self.output_layer:
            elem.insert_in_annetto()
            rdfWrapper.new_output_layer(self.name,elem.name)
        #rdfWrapper.new_output_layer(self.name,self.input_layer.name)


    def __init__(self,name):
        self.name=name
        self.type="Network"
        self.layer = {}
        self.metric = {}
        self.activation = {}
        self.output = {}
        self.loss_function = {}
        self.input_layer=[]
        self.output_layer=[]
        self.optimizer={}
        self.objective={}




        self.task_type={}