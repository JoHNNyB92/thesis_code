import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from functions.activation.non_diff.relu import relu
from functions.activation.regularization.dropout import dropout
from functions.activation.smooth.softmax import softmax
from functions.activation.smooth.tanh import tanh
from functions.activation.smooth.sigmoid import sigmoid
class layer:

    def insert_in_annetto(self):
        print("Annetto::layer-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        if self.activation!=None:
            self.activation.insert_in_annetto()
            rdfWrapper.new_has_activation(self.name,self.activation.name)
        if self.hasBias==True:
            rdfWrapper.new_has_bias(self.name,self.hasBias)
        if self.num_layer=="":
            print("ERROR:"+self.name+" NOT AVAILABLE NUM LAYER")
        else:
            rdfWrapper.layer_num_units(self.name, self.num_layer)
        print("self=",self.name)
        for elem in self.next_layer:
            rdfWrapper.new_next_layer(self.name, elem)
        for elem in self.previous_layer:
            rdfWrapper.new_previous_layer(self.name, elem)

    def update_dicts(self):
        nodes.handler.entitiesHandler.layers_to_nodes[self.name]=[]
        for elem in self.output_nodes:
            nodes.handler.entitiesHandler.nodes_to_layers[elem]=self.name
            nodes.handler.entitiesHandler.layers_to_nodes[self.name].append(elem)
        for elem in self.input:
            nodes.handler.entitiesHandler.nodes_to_layers[elem]=self.name
            nodes.handler.entitiesHandler.layers_to_nodes[self.name].append(elem)
    '''
    def check_if_in(self,name):
        prevRes=[]
        for input in nodes.handler.entitiesHandler.node_map[name].get_inputs():
            print("NODE:IN----",name,":",input.get_name())
            if input.get_name() not in nodes.handler.entitiesHandler.layer_names:
                if input.get_op()=="Placeholder":
                    print("RETURNING=",input.get_name(),":",input.get_op())
                else:
                    print(input.get_name(),":input.get=",input.get_op())
                    res=self.check_if_in(input.get_name())
                    if res==False:
                        print("IT IS FALSE FOR =",input.get_name())
                        return res
            else:
                print("LAYER INPUT ",name,":",input.get_name()," IS FALSE")
                return False
    '''
    def init_activation(self,node):
        if node.get_op()=="Relu":
            self.activation=relu(node)
        elif node.get_op()=="Dropout":
            self.activation=dropout(node)
        elif node.get_op()=="Softmax":
            self.activation = softmax(node)
        elif node.get_op()=="Tanh":
            self.activation = tanh(node)
        elif node.get_op()=="Sigmoid":
            self.activation = sigmoid(node)
        else:
            print("NOT HANDLED ACTIVATION=",node.get_op())
            import sys
            sys.exit()

    def find_output_node(self,name):
        print("FINDING OUTPUT FOR ",name)
        nm = nodes.handler.entitiesHandler.node_map
        for node_name in nm.keys():
            #print("Name=",node_name," result ",nm[node_name].search_inputs(name))
            if nm[node_name].search_inputs(name) == True and "gradient" not in node_name:
                print("SELF=",name," || SELF IN INPUT OF =",node_name,"{}",[x.get_name() for x in nodes.handler.entitiesHandler.node_map[node_name].get_inputs()])
                #Problem with reshape,it is an in+
                # ermediate operation,though when used with flatten there is a problem.
                #TODO:FIND A BETTER SOLUTION-Insert all intermediate nodes into a temporary intermediate storage,aftwrwards check this when connecting the layers

                if nm[node_name].get_op() in nodes.handler.entitiesHandler.intermediate_operations and "(" not in node_name.split("/")[-1]:
                    print(self.name,":1:", node_name)
                    self.output_intermediate_nodes[node_name]=name
                    self.find_output_node(node_name)
                elif nm[node_name].get_op() in nodes.handler.entitiesHandler.activation_operations:
                    self.init_activation(nm[node_name])
                    self.find_output_node(node_name)
                else:
                    print(self.name,":3:", node_name)
                    if self.output_intermediate_nodes.keys()!=[]:
                        self.output_intermediate_nodes[node_name]=name
                    if node_name not in self.output_nodes:
                        self.output_nodes.append(node_name)
        if self.activation=="":
            for node_name in nm.keys():
                if nm[node_name].search_inputs(self.name) == True:
                    print("INPUT NODE RE MALAKA =",node_name)
                    if nm[node_name].get_op() in nodes.handler.entitiesHandler.activation_operations:
                        self.init_activation(nm[node_name])
                        return

    def find_output_with_activation(self,name):
        nm = nodes.handler.entitiesHandler.node_map
        for node_name in nodes.handler.entitiesHandler.node_map.keys():
            if nm[node_name].search_inputs(name) == True:
                print("ELELELELEL=",name)
                if "gradient" not in nm[node_name].get_name():
                    print("ELELELELEL2=", name)
                    if nm[node_name].get_op() in nodes.handler.entitiesHandler.activation_operations or \
                            nm[node_name].get_op() in nodes.handler.entitiesHandler.intermediate_operations:
                        print("about to call=",node_name)
                        self.find_output_with_activation(node_name)
                        return
                    else:
                        print("output is =",node_name)
                        self.output_nodes.append(node_name)

    def find_input_layer(self,input_node):
        print("FINDING INPUT LAYER FOR ",self.name)
        print("INPUT NODES ARE ",self.input)
        for input_name in set(self.input):
            if input_name!=self.name:
                if input_name in nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer.keys() and input_name!=self.name :
                    print("Immediate connection found between ", input_name, " and ", self.name)
                    input_node=nodes.handler.entitiesHandler.node_map[input_name]
                    self.previous_layer.append(input_node.get_name())
                    nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[input_node.get_name()].next_layer.append(self.node.get_name())
        for layer in nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer.keys():
            layer_obj = nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[
                layer]
            elems_in_both_lists = set(layer_obj.output_nodes) & set(self.input)
            print("Check if output of ",layer_obj.name," and ",self.name," input have common elements.The result is ",elems_in_both_lists)
            if layer_obj.name!= self.name and (self.name in layer_obj.output_nodes or  len(elems_in_both_lists)!=0):
                found_in_other_layer=False
                for elem in elems_in_both_lists:
                    #print("TEST:ELEM=",elem," BOTH=",elems_in_both_lists)
                    #for key in layer_obj.output_intermediate_nodes:
                        #print("Key=",key," Intermediate=",layer_obj.output_intermediate_nodes)
                    if elem in layer_obj.output_intermediate_nodes:
                        #print("TEST:",elem)
                        temp=layer_obj.output_intermediate_nodes[elem]
                        while temp in layer_obj.output_intermediate_nodes:
                            for layer in nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer.keys():
                                '''if nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[layer].inner_nodes!=[]:
                                    print(nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[layer].inner_nodes)
                                else:
                                    print("layer=",layer," does not have innert nodes ")
                                '''
                                if temp in nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[layer].inner_nodes:
                                    print("INSIDEEEEEEEEEEEEEEEEEEEEEEEEE")
                                    found_in_other_layer=True
                                    temp=""
                            if temp in layer_obj.output_intermediate_nodes:
                                temp=layer_obj.output_intermediate_nodes[temp]
                            else:
                                temp=""
                if found_in_other_layer==False:
                    #Case of already added from above case
                    if layer_obj.name not in self.previous_layer:
                        self.previous_layer.append(layer_obj.name)
                        nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[
                            layer_obj.name].next_layer.append(self.name)
                        print("1)NODE:", self.node.get_name(), "\nPL:", self.previous_layer, "\nLAY:", layer, "\nNL:",
                              nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[
                                  layer].next_layer)
                    #return
                    else:
                        for output_node in  layer_obj.output_nodes:
                            for input_name in self.input:
                                input_node = nodes.handler.entitiesHandler.node_map[input_name]
                                if input_node.search_inputs(output_node) == True:
                                    self.previous_layer.append(layer_obj.name)
                                    nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[
                                        layer].next_layer.append(self.node.get_name())
                                    print("2)NODE:", self.node.get_name(), " PL:", self.previous_layer, " LAY:", layer, " NL:",
                                          nodes.handler.entitiesHandler.data.annConfiguration.networks[nodes.handler.entitiesHandler.current_network].layer[
                                              layer].next_layer[0])
                                    return
                else:
                    print("Some element of ",elems_in_both_lists," is inner element of a layer.")
                    # handle intermediate transformations(unstack etc)
        print("FINDING INPUT LAYER RESULT ", self.name, " - ", self.previous_layer)

    def check_input_layer(self,name):
        res=self.check_in(name)
        if res==False:
            self.is_in=False
        else:
            self.is_in=True

    def find_output(self, nn):
        print("Find output=")
        for tmp in nodes.handler.entitiesHandler.node_map.keys():
            print("Find output=",tmp)
            if nodes.handler.entitiesHandler.node_map[tmp].search_inputs(nn) == True:
                self.next_layer.append(tmp)
                return

    def find_num_layers(self):
        for elem in self.node.get_output():
            for num in elem.dim:
                if num.size > 0:
                    self.num_layer = num.size
                    break
    def get_all_inner_nodes(self):
        nm = nodes.handler.entitiesHandler.node_map
        for name in nm.keys():
            if self.name+"/" in name:
                self.inner_nodes.append(name)

    def find_input_node_complex(self):
        nm = nodes.handler.entitiesHandler.node_map
        for node in nodes.handler.entitiesHandler.node_map.keys():
            elem = nodes.handler.entitiesHandler.node_map[node]
            if self.name in elem.get_name() and "gradient" not in elem.get_name().lower():
                for elem_in in elem.get_inputs():
                    if self.name not in elem_in.get_name():
                        res = 0
                        for elem in nm[node].get_inputs():
                            if self.name not in elem.get_name() and "gradients" not in elem.get_name():
                                res += 1
                        if res != 0:
                            found = False
                            input = nm[node].get_inputs()
                            print("1:START SEARCHING FOR=", node)
                            while found == False:
                                temp_in = []
                                for elem_in in input:
                                    print("2:START SEARCHING FOR=", elem_in.get_name())
                                    if elem_in.get_op() in nodes.handler.entitiesHandler.intermediate_operations:
                                        for elem_in_in in nm[elem_in.get_name()].get_inputs():
                                            temp_in.append(elem_in_in)
                                    else:
                                        found = True
                                        self.input.append(node)
                                        break
                                input = temp_in

    def find_output_node_complex(self):
        for name in nodes.handler.entitiesHandler.node_map.keys():
            elem = nodes.handler.entitiesHandler.node_map[name]
            if self.name not in elem.get_name() and "gradient" not in elem.get_name().lower():
                for elem_in in elem.get_inputs():
                    if self.name in elem_in.get_name():
                        if elem.get_op() in nodes.handler.entitiesHandler.intermediate_operations or elem.get_op() in nodes.handler.entitiesHandler.activation_operations:
                            print("2FUCKING:",elem.get_name())
                            self.find_output_node(elem.get_name())
                        elif elem.get_name() not in self.output_nodes:
                            print("3FUCKING:", elem.get_name())
                            self.output_nodes.append(elem.get_name())

    def __init__(self,node,name,hasBias,searchForInner=False):
        self.node=node
        self.name=name
        self.hasBias=hasBias
        self.num_layer=""
        self.find_num_layers()
        self.activation=None
        self.input=[]
        self.next_layer=[]
        self.previous_layer=[]
        self.inner_nodes=[]
        self.output_intermediate_nodes={}
        if searchForInner==True:
            self.get_all_inner_nodes()


