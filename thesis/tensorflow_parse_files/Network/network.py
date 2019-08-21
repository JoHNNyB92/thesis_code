import virtuosoWrapper.virtuosoWrapper as rdfWrapper
class network:
    def insert_in_annetto_netwr(self):
        rdfWrapper.new_named_individual(self.name)
        temp_dict_layer=self.layer.copy()
        for elem in temp_dict_layer.keys():
            for elem_out in self.output_layer:
                if elem_out.name==elem and elem in self.layer.keys():
                    del self.layer[elem]
        for layer_name in self.layer.keys():
            self.layer[layer_name].insert_in_annetto()
            rdfWrapper.new_network_has_layer(self.name, layer_name)
        if self.objective!="":
            for objective_name in self.objective.keys():
                self.objective[objective_name].insert_in_annetto()
                rdfWrapper.new_network_has_objective(self.name, objective_name)

        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_network(self.name)
        for elem in self.input_layer:
            elem.insert_in_annetto()
            rdfWrapper.new_input_layer(self.name,elem.name)
        out_inserted=[]
        for elem in self.output_layer:
            if elem.name not in out_inserted:
                out_inserted.append(elem.name)
                elem.insert_in_annetto()
                rdfWrapper.new_output_layer(self.name,elem.name)
        for elem in self.hang_dp:
            elem.insert_in_annetto()

    def __init__(self,name):
        self.name=name
        self.datasets={}
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
        self.hang_dp=[]
        self.task_type={}