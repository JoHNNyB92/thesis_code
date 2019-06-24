import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from layers.layer import layer
class in_out_layer(layer):

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        if self.num_layer!="":
            rdfWrapper.layer_num_units(self.name, self.num_layer)
        else:
            print("ERROR:IN OUT W/O NUM LAYER")

    def find_output_node(self,name):
        layer.find_output_node(self,name)

    def find_input_layer(self):
        #print("FINDING INPUT LAYER OF IN/OUT=",self.name)
        layer.find_input_layer(self,self.node)

    def __init__(self,node):
        super(in_out_layer, self).__init__(node, node.get_name(), False)
        self.output_nodes = []
        #self.type="Placeholder"
        self.type="InOutLayer"
        self.node=node
        self.name=node.get_name()
        self.find_output_node(self.name)