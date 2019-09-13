from functions.loss.loss import loss
import nodes.handler

class binary_cross_entropy(loss):

    def insert_in_annetto(self):
        super(binary_cross_entropy, self).insert_in_annetto()

    def get_all_inner_nodes(self):
        super(binary_cross_entropy, self).get_all_inner_nodes()

    def __init__(self, name,node):
        super(binary_cross_entropy, self).__init__(name, node)
        self.type = "BinaryCrossEntropy"
