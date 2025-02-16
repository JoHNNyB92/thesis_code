import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class dataset_pipe:

    def insert_in_annetto(self):
        res=rdfWrapper.new_named_individual(self.name)
        if res==0:
            rdfWrapper.new_type(self.name, self.type)
            rdfWrapper.new_joins_layer(self.name,self.input_layer.name)
            if self.dataset!="":
                self.dataset.insert_in_annetto()
                rdfWrapper.new_joins_dataset(self.name,self.dataset.name)

    def __init__(self,node,layer,type,count,dataset):
        self.input_layer=layer
        self.dataset=dataset
        if type=="train":
            self.name=node.get_name()+"_INP_"+count
        else:
            #self.find_datasets(node)
            self.name = node.get_name() + "_EVP_"+count
        self.type = "DatasetPipe"
        self.node=node

    def find_datasets(self,helper):
        found = False
        children = [x.get_name() for x in helper.get_inputs()]
        test_found = False
        while found == False:
            temp_new_children = []
            for elem in children:
                if nodes.handler.entitiesHandler.node_map[elem].get_op() == "Placeholder" and test_found == False:
                    found = True
                    #self.input_layer = elem
                    print("YOLELELELEL=",elem)
                    self.dataset=elem
                else:
                    for x in nodes.handler.entitiesHandler.node_map[elem].get_inputs():
                        temp_new_children.append(x.get_name())
            children = []
            for x in temp_new_children:
                children.append(x)