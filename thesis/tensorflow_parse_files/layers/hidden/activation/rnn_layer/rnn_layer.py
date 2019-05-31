from layers.layer import layer

class rnn_layer(layer):

    def get_all_inner_nodes(self):
        super(rnn_layer,self).get_all_inner_nodes()

    def find_input_node_complex(self):
        super(rnn_layer,self).find_input_node_complex()

    def find_output_node_complex(self):
        super(rnn_layer,self).find_output_node_complex()

    def update_dicts(self):
        super(rnn_layer, self).update_dicts()

    def insert_in_annetto(self):
        super(rnn_layer, self).insert_in_annetto()

    def __init__(self,node,name,hasBias):
        super(rnn_layer, self).__init__(node,name,hasBias)
