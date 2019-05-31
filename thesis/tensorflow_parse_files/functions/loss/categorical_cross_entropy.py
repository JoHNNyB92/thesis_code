from functions.loss.loss import loss
import nodes.handler
class categorical_cross_entropy(loss):

    def insert_in_annetto(self):
        super(categorical_cross_entropy, self).insert_in_annetto()

    def get_all_inner_nodes(self):
        nm = nodes.handler.entitiesHandler.node_map
        for name in nm.keys():
            if self.name+"/" in name:
                self.inner_nodes.append(name)

    def __init__(self,name,node):
        super(categorical_cross_entropy, self).__init__(name, node)
        self.type="categorical"
