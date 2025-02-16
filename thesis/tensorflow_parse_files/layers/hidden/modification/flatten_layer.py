from layers.layer import layer
import nodes.handler

class flatten_layer(layer):

    def find_input_node_complex(self):
        super(flatten_layer,self).find_input_node_complex()

    def find_output_node_complex(self):
        super(flatten_layer,self).find_output_node_complex()

    def update_dicts(self):
        super(flatten_layer, self).update_dicts()

    def find_input_layer(self):
        layer.find_input_layer(self, self.node)

    def find_output_node(self,name):
        layer.find_output_node(self,name)

    def __init__(self,node,name):
        self.type="FlattenLayer"
        self.output_nodes = []
        super(flatten_layer, self).__init__(node, name,False,True)
        self.find_input_node_complex()
        self.find_output_node_complex()
        self.update_dicts()
