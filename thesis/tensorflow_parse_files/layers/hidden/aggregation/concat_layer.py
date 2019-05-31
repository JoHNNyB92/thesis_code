from layers.layer import layer


class concat_layer(layer):

    def update_dicts(self):
        super(concat_layer, self).update_dicts()

    def find_input_layer(self):
        layer.find_input_layer(self, self.node.get_inputs()[0])

    def find_output_node(self, name):
        layer.find_output_node(self, name)

    def __init__(self, node):
        super(concat_layer, self).__init__(node, node.get_name(), False)
        self.type = "ConcatLayer"
        self.output_nodes = []
        self.find_output_node(self.name)
        self.update_dicts()
