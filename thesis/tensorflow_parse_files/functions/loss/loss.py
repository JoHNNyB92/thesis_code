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

    def find_input_nodes(self):
        nm=nodes.handler.entitiesHandler.node_map
        for node in nm.keys():
            if "gradient" not in nm[node].get_name() and self.name in nm[node].get_name().split("/"):
                for elem in nm[node].get_inputs():
                    if self.name not in elem.get_name():
                        self.input_nodes.append(elem)
                        break



    def __init__(self,name,node,complex=False):
        super(loss, self).__init__(name,node)
        self.inner_nodes=[]
        self.input_nodes=[]
        if complex==True:
            self.find_input_nodes()
        self.get_all_inner_nodes()