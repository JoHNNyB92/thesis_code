from functions.loss.loss import loss
import nodes.handler

class categorical_cross_entropy(loss):

    def insert_in_annetto(self):
        super(categorical_cross_entropy, self).insert_in_annetto()

    def get_all_inner_nodes(self):
        super(categorical_cross_entropy, self).get_all_inner_nodes()

    def __init__(self,name,node,complex=False):
        super(categorical_cross_entropy, self).__init__(name, node,complex)
        self.type="CategoricalCrossEntropy"
