from layers.layer import layer
import nodes.handler

class dropout_layer(layer):

    def insert_in_annetto(self):
        super(dropout_layer, self).insert_in_annetto()

    def find_input_node_complex(self):
        super(dropout_layer,self).find_input_node_complex()

    def find_output_node_complex(self):
        super(dropout_layer,self).find_output_node_complex()

    def update_dicts(self):
        super(dropout_layer, self).update_dicts()

    def find_num_layers(self):
        layer.find_num_layers(self)

    def find_input_layer(self):
        layer.find_input_layer(self, self.node)

    def find_output_node(self, name):
        layer.find_output_node(self, name)

    def __init__(self, node, name):
        self.type = "DropoutLayer"
        self.output_nodes = []
        super(dropout_layer, self).__init__(node, name, False, True)
        self.find_input_node_complex()
        self.find_output_node_complex()
        self.update_dicts()
        #print("LOGGING:CLASS INFORMATION:",self.type, ":Name:", self.name, "\nInput Nodes:", set(self.input), "\nOutput Nodes:", set(self.output_nodes))

