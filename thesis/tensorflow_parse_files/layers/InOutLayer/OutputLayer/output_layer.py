from layers.InOutLayer.in_out_layer import in_out_layer
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler

class output_layer(in_out_layer):

    def insert_in_annetto(self):
        #print("Annetto::output_layer-", self.name)
        super(output_layer, self).insert_in_annetto()
        for elem in self.previous_layer:
            rdfWrapper.new_previous_layer(self.name, elem)
        #rdfWrapper.new_previous_layer(self.name, self.previous_layer)

    def find_num_layers(self):
        for elem in self.node.get_output():
            for num in elem.dim:
                if int(num.size) > 0:
                    self.num_layer = num.size
                    break

    def __init__(self,layer):
        super(output_layer, self).__init__(layer.node)
        self.previous_layer=layer.previous_layer
        self.orig_layer=layer
        self.next_layer = []
        self.num_layer=""
        self.find_num_layers()
        self.type = "OutputLayer"
