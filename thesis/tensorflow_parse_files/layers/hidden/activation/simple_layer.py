from layers.layer import layer
import nodes.handler

class simple_layer(layer):

    def update_dicts(self):
        super(simple_layer, self).update_dicts()

    def insert_in_annetto(self):
        super(simple_layer, self).insert_in_annetto()

    def find_input(self):
        nodes.handler.entitiesHandler.all_inputs[self.matMul.get_name()] = self.node.get_name()

    def find_input_layer(self):
        layer.find_input_layer(self,self.matMul)

    def find_output_node(self,name):
        layer.find_output_node(self,name)

    def __init__(self,W,X,MM,B,node,name=""):
        if name=="":
            super(simple_layer, self).__init__(node,node.get_name(),True)
        else:
            super(simple_layer, self).__init__(node, name, True)
        self.W=W
        self.X=X
        self.B=B
        self.output_nodes = []
        self.find_output_node(self.name)
        self.matMul=MM
        self.type="FullyConnectedLayer"
        self.input.append(self.matMul.get_name())
        self.next_layer=[]
        self.update_dicts()



