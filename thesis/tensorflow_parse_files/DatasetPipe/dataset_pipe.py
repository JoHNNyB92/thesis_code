import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class dataset_pipe:

    def insert_in_annetto(self):
        #print("Annetto::dataset_pipe-", self.name)
        inputLayer=self.node.name
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_joins_layer(self.name,inputLayer)
        rdfWrapper.new_joins_dataset(self.name,self.joins_dataset)

    def __init__(self,name,node,helper,type,count):
        self.input_layer=""
        if type=="train":
            self.name=name+"_INP_"+count
        else:
            self.name = name + "_EVP_"+count
        self.type = "DatasetPipe"
        self.joins_dataset=""
        self.node=node
        self.find_datasets(type,helper)


    def find_datasets(self,type,helper):
        found = False
        children = [x.get_name() for x in helper.get_inputs()]
        test_found = False
        while found == False:
            temp_new_children = []
            for elem in children:
                if nodes.handler.entitiesHandler.node_map[elem].get_op() == "Placeholder" and test_found == False:
                    found = True
                    self.input_layer = elem
                    self.joins_dataset=elem
                    #print("FOUND ",type," DATASET=", self.joins_dataset)
                else:
                    for x in nodes.handler.entitiesHandler.node_map[elem].get_inputs():
                        temp_new_children.append(x.get_name())
            children = []
            for x in temp_new_children:
                children.append(x)