from layers.layer import layer
from  layers.hidden.activation.rnn_layer.rnn_layer import rnn_layer
import nodes.handler

class gru_layer(rnn_layer):

    def get_all_inner_nodes(self):
        super(gru_layer,self).get_all_inner_nodes()

    def find_input_node_complex(self):
        super(gru_layer,self).find_input_node_complex()

    def find_output_node_complex(self):
        super(gru_layer,self).find_output_node_complex()

    def insert_in_annetto(self):
        super(gru_layer, self).insert_in_annetto()

    def find_input_layer(self):
        #layer.find_input_layer(gru_layer,self.input)
        super(gru_layer, self).find_input_layer(self.input)

    def find_output_node(self,name):
        super(gru_layer, self).find_output_node(name)

    def update_dicts(self):
        super(gru_layer, self).update_dicts()

    def __init__(self,node,name):
        super(gru_layer, self).__init__(node,name,False)
        self.type="GRU"
        self.name=name
        self.output_nodes = []
        self.find_input_node_complex()
        self.find_output_node_complex()
        self.get_all_inner_nodes()
        self.update_dicts()