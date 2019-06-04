from layers.layer import layer
from  layers.hidden.activation.rnn_layer.rnn_layer import rnn_layer
import nodes.handler
class lstm_layer(rnn_layer):

    def get_all_inner_nodes(self):
        super(lstm_layer,self).get_all_inner_nodes()

    def find_input_node_complex(self):
        super(lstm_layer,self).find_input_node_complex()

    def find_output_node_complex(self):
        super(lstm_layer,self).find_output_node_complex()

    def insert_in_annetto(self):
        super(lstm_layer, self).insert_in_annetto()

    def find_input_layer(self):
        layer.find_input_layer(self,self.input)

    def find_output_node(self,name):
        super(lstm_layer, self).find_output_node(name)

    def update_dicts(self):
        super(lstm_layer, self).update_dicts()

    def __init__(self,node,name):
        super(lstm_layer, self).__init__(node,name,False)
        self.type="LSTMLayer"
        self.output_nodes = []
        self.find_input_node_complex()
        self.find_output_node_complex()
        self.get_all_inner_nodes()
        self.update_dicts()
        print("LOGGING:CLASS INFORMATION:", self.type, ":Name:", self.name, "\nInput Nodes:", self.input, "\nOutput Nodes:",self.output_nodes)