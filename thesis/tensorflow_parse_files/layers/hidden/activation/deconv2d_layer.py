from layers.layer import layer
import nodes.handler

class deconv2d_layer(layer):

    def update_dicts(self):
        super(deconv2d_layer, self).update_dicts()

    def insert_in_annetto(self):
        super(deconv2d_layer, self).insert_in_annetto()

    def insert_in_annetto(self):
        super(deconv2d_layer, self).insert_in_annetto()

    def find_input_layer(self):
        layer.find_input_layer(self, self.kernel)

    def find_output_node(self, name):
        layer.find_output_node(self, name)

    def __init__(self,input,kernel,node,biasAdd):
        super(deconv2d_layer, self).__init__(node, node.get_name(),True)
        self.input_layer=input
        self.kernel=kernel
        self.type="DeconvolutionLayer"
        self.biasadd=biasAdd
        self.output_nodes = []
        self.find_output_node(self.biasadd)
        self.dense_layer_name = ""
        self.update_dicts()


