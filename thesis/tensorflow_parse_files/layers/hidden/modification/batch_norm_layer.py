from layers.layer import layer
import nodes.handler

class batch_norm_layer(layer):

    def insert_in_annetto(self):
        super(batch_norm_layer, self).insert_in_annetto()

    def update_dicts(self):
        super(batch_norm_layer, self).update_dicts()

    def find_input_layer(self):
        layer.find_input_layer(self, self.node)

    def find_output_node(self, name):
        layer.find_output_node(self, name)

    def __init__(self, node, name):
        self.type = "BatchNormLayer"
        self.output_nodes = []
        super(batch_norm_layer, self).__init__(node, name, False, True)
        self.find_output_node(self.name)
        self.update_dicts()
        #No reason to search for input nodes
        #print("LOGGING:CLASS INFORMATION:",self.type, ":Name:", self.name, "\nInput Nodes:", set(self.input), "\nOutput Nodes:", set(self.output_nodes))

