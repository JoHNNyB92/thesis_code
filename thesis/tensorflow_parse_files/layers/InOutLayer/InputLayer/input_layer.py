from layers.InOutLayer.in_out_layer import in_out_layer
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class input_layer(in_out_layer):

    def insert_in_annetto(self):
        #print("Annetto::input_layer-", self.name)
        super(input_layer, self).insert_in_annetto()
        for elem in self.next_layer:
            rdfWrapper.new_next_layer(self.name, elem)

    def find_num_layers(self):
        for elem in self.node.get_output():
            for num in elem.dim:
                if num.size > 0:
                    self.num_layer = num.size
                    break

    def __init__(self,node):
        super(input_layer, self).__init__(node)
        self.num_layer=""
        self.find_num_layers()
        self.next_layer=[]
        self.previous_layer="NONE"
        self.type="InputLayer"
        #print("CLASS INFORMATION:",self.type, ":Name:", self.name, "\nInput Nodes:", set(self.input), "\nOutput Nodes:", set(self.output_nodes))

