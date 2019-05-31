from functions.function import function
import nodes.handler
class loss(function):

    def get_all_inner_nodes(self):
        nm = nodes.handler.entitiesHandler.node_map
        for name in nm.keys():
            if self.name+"/" in name:
                self.inner_nodes.append(name)

    def insert_in_annetto(self):
        super(loss, self).insert_in_annetto()

    def __init__(self,name,node):
        super(loss, self).__init__(name,node)
        self.inner_nodes=[]
        print("TROUBOUKI=",self.name)
        self.get_all_inner_nodes()