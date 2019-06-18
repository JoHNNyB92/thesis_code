from virtuosoWrapper.annet_o_data_management import annet_o_data_management as dataMgnt
from NetworkEvaluation.evaluation_result import evaluation_result
from nodes import handler_functions
from nodes.nodeEntity import nodeEntity as nodeClass

class handle_entities:
    def __init__(self):
        self.data=dataMgnt()
        self.batch=0
        self.epochs=0
        self.node_map={}
        self.current_network=""
        #self.networks=[]
        #self.current_evaluation=""
        #self.activations={}
        #self.all_inputs={}
        #self.intermediate_bias={}
        self.possible_loss_function={}
        self.variable_operations=["Const","VariableV2","Variable"]

        #Operations that must be ignored if presented into position that do not represent something important.
        self.intermediate_operations=["Add","Identity","Unpack","Reshape","StridedSlice","Range","mul","GatherV2","Pack","Transpose","concat","Mean","ExpandDims","Fill","Ones","Tile"]
        self.optimizer_operations=["ApplyAdam"]
        #Activation functions upported as of now
        self.activation_operations=["Elu","Relu","Dropout","Softmax","Sigmoid","Tanh","Softplus"]
        self.loss_functions=["softmax_cross_entropy"]
        #Sometimes the layers are part of a wider node system,thus we want to know if we encounter a node whether it is part of a
        #layer.
        self.nodes_to_layers={}
        #[COMM]
        self.layers_to_nodes = {}
        self.regularizers=["L2Loss"]
        self.loss=["Equal","Mean"]
        self.batch=""
        self.epoch=""
        self.discovered_loss = []
        self.discovered_optimizers = []
        self.input_layers = []
        self.names_into_separator=[]
        self.output_layer=[]
        self.input_layer=[]

    def set_batch_epoch(self,batch,epoch):
        #Set batch and epoch,though program has some limitations due to lack of gpu.
        self.batch=batch
        self.epoch=epoch

    #The following function is used only if there is only one network presented.
    def insert_to_evaluation_pipe(self):
        for layer in self.data.annConfiguration.networks[self.current_network].layer.keys():
            #If next layer is empty,it means we encounter a node just before the loss function.Thus we can search for
            if self.data.annConfiguration.networks[self.current_network].layer[layer].next_layer==[]:
                #[COMM]
                for i,_ in enumerate(self.output_layer):
                    self.output_layer[i].previous_layer.append(layer)
                    self.data.annConfiguration.networks[self.current_network].layer[layer].next_layer.append(self.output_layer[i].name)
                break
        # [COMM]
        IOPipe=handler_functions.handle_dataset_pipe(self.current_network,self.output_layer,"test")
        self.data.evaluationResult.IOPipe=IOPipe
        self.data.evaluationResult.ann_conf=self.data.annConfiguration

    #If some prerequisites are satisfied,we are into the basic handling of the result of the node encountered.
    def insert_to_list(self,node,name,case):
        if case=="Layer":
            print("LOGGING: Layer: ",name," Type: ",node.type)
            self.data.annConfiguration.networks[self.current_network].layer[name]=node
        elif case=="Optimizer":
            print("LOGGING: Optimizer: ", name, " Type: ", node.type)
            self.data.annConfiguration.networks[self.current_network].optimizer[name]=node
        elif case=="Objective":
            print("LOGGING: Objective: ", name, " Type: ", node.type)
            self.data.annConfiguration.networks[self.current_network].objective[name] = node

    #In order to extract the name of the layer it requires the extraction of the name based on some bigger node
    #it may be part of .
    def handle_complex_layers(self,node,name,case):
        #Split the name based on / and when the case (rnn,dense etc) is part of the splitted word,this is the name.
        tmp = name.split("/")
        tmp_name = ""
        for x in tmp:
            if case in x:
                tmp_name = tmp_name + "/" + x
                break
            else:
                tmp_name = tmp_name + "/" + x
        #Exclude first /
        tmp_name = tmp_name[1:]
        #If this was not encountered again,insert into list keeping complex node names.
        if tmp_name not in self.names_into_separator:
            self.names_into_separator.append(tmp_name)
            if case=="dense":
                    (isSimpleLayer, nodeReturn) = handler_functions.check_simple_layer(self.node_map[tmp_name + "/BiasAdd"],
                                                                                       tmp_name + "/BiasAdd")
                    if isSimpleLayer == True:
                        self.insert_to_list(nodeReturn, tmp_name + "/BiasAdd", "Layer")
            elif case =="dropout":
                nodeReturn = handler_functions.handle_dropout(node, tmp_name)
                self.insert_to_list(nodeReturn, tmp_name, "Layer")
            elif case=="flatten":
                nodeReturn = handler_functions.handle_flatten(node, tmp_name)
                self.insert_to_list(nodeReturn, tmp_name, "Layer")
            elif case == "rnn":
                nodeReturn = handler_functions.handle_lstm(node,tmp_name)
                self.insert_to_list(nodeReturn, tmp_name, "Layer")

    def optimizers(self,keyword,e):
        name = self.node_map[e].get_name().split("/")
        real_name = ""
        for part in name:
            if keyword in part:
                real_name = part
                break
        if real_name not in self.discovered_optimizers:  # and self.node_map[e].get_name().endswith("learning_rate"):
            self.discovered_optimizers.append(real_name)
            return (True,real_name)
        return (False,"")

    def objectives(self,c_function,e):
        name = "objective_function"
        if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
            name = name + "_" + str(len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
        self.discovered_loss.append(self.node_map[e].get_name().split("/")[0])
        nodeReturn = handler_functions.handle_objective(name, c_function)
        if nodeReturn != "":
            self.insert_to_list(nodeReturn, name, "Objective")
    '''
    elif self.node_map[e].get_name().startswith("RMSProp") and self.optimizer==False:
        self.optimizer = True
        nodeReturn = handler_functions.handle_rms_prop(self.node_map[e])
        self.insert_to_list(nodeReturn, e,"Optimizer")
    elif self.node_map[e].get_name().startswith("GradientDescent") and self.optimizer==False:
        self.optimizer = True
        nodeReturn = handler_functions.handle_gradient_descent(self.node_map[e])
        self.insert_to_list(nodeReturn, e,"Optimizer")
    '''
    #Main function, used for handling each protocol buffer node produced by the executed
    def handle_different_layer_cases(self,e):
        # If gradient is part of name,we ignore it,due to the fact it complicates the process of understanding the nn architecture
        if "gradient" in e:
            return
        #If we encounter Add ,it might be possible a layer case.
        if self.node_map[e].get_op()=="Add":
            (isSimpleLayer,nodeReturn)=handler_functions.check_simple_layer(self.node_map[e],self.node_map[e].get_name())
            if isSimpleLayer==True:
                self.insert_to_list(nodeReturn,e,"Layer")
        elif "BatchNorm" in self.node_map[e].get_op():
            nodeReturn = handler_functions.handle_batch_norm(self.node_map[e])
            self.insert_to_list(nodeReturn, e, "Layer")
        elif self.node_map[e].get_op() == "ConcatV2":
            nodeReturn = handler_functions.handle_concat(self.node_map[e])
            self.insert_to_list(nodeReturn, e, "Layer")
        elif self.node_map[e].get_op()=="MaxPool" or self.node_map[e].get_op()=="AvgPool":
            nodeReturn=handler_functions.handle_maxpool(self.node_map[e])
            self.insert_to_list(nodeReturn, e,"Layer")
        elif self.node_map[e].get_op()=="Conv2D":
            nodeReturn= handler_functions.handle_conv2d(self.node_map[e])
            if nodeReturn!=None:
                self.insert_to_list(nodeReturn, e,"Layer")
        elif self.node_map[e].get_op()=="Conv2DBackpropInput" :
            nodeReturn= handler_functions.handle_deconv2d(self.node_map[e])
            if nodeReturn!=None:
                self.insert_to_list(nodeReturn, e,"Layer")
        elif "RMSProp" in self.node_map[e].get_name():
            (res, real_name) = self.optimizers("RMSProp", e)
            if res == True:
                nodeReturn = handler_functions.handle_rms_prophandle_rms_prop(self.node_map[e], real_name)
                self.insert_to_list(nodeReturn, real_name, "Optimizer")
        elif "GradientDescent" in self.node_map[e].get_name():
            (res, real_name) = self.optimizers("GradientDescent", e)
            if res == True:
                nodeReturn = handler_functions.handle_gradient_descent(self.node_map[e], real_name)
                self.insert_to_list(nodeReturn, real_name, "Optimizer")
        elif "Adam" in self.node_map[e].get_name():
            (res,real_name)=self.optimizers("Adam",e)
            if res==True:
                nodeReturn = handler_functions.handle_adam(self.node_map[e],real_name)
                self.insert_to_list(nodeReturn, real_name,"Optimizer")
        #Neg node might be part of a categorical cross entropy created by the user using custom functions
        #and not the tensorflow implemented function.
        elif self.node_map[e].get_op()=="Neg":
            nodeReturn = handler_functions.handle_neg_for_log(self.node_map[e])
            if nodeReturn!="":
                self.objectives(nodeReturn,e)
        elif self.node_map[e].get_op()=="Mul":
            c_name = "cost_function"
            num="0"
            if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
                num = str(len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
            c_name = c_name + "_" + num
            nodeReturn=handler_functions.handle_mul_as_cross_entropy(self.node_map[e],c_name)
            if nodeReturn!="":
                self.objectives(nodeReturn,e)
        elif self.node_map[e].get_op()=="SparseSoftmaxCrossEntropyWithLogits":
            c_name="cost_function"
            # TODO:NEED TO DECIDE WHAT TO DO WITH MIN/MAX,RIGHT NOW by default min
            if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
                num=str(len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
                c_name=c_name+"_"+num
            nodeReturn = handler_functions.handle_sparse_cross_entropy(self.node_map[e],c_name)
            if nodeReturn != "":
                self.objectives(nodeReturn,e)
        elif "sigmoid_cross_entropy" in e or "logistic_loss" in e:
            parts=e.split("/")
            full_name=""
            for part in parts:
                full_name = full_name + "/" + part
                if "sigmoid_cross_entropy" or "logistic_loss" in part:
                    break
            full_name=full_name[1:]
            if full_name not in self.names_into_separator:
                self.names_into_separator.append(full_name)
                c_name="cost_function"
                # TODO:NEED TO DECIDE WHAT TO DO WITH MIN/MAX,RIGHT NOW by default min
                if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
                    num=str(len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
                    c_name=c_name+"_"+num
                nodeReturn = handler_functions.handle_sigmoid_entropy(self.node_map[e],c_name,full_name)
                if nodeReturn != "":
                    self.objectives(nodeReturn, e)
        elif self.node_map[e].get_op()=="SoftmaxCrossEntropyWithLogits":
            name_loss = e
            c_name = "cost_function"
            # TODO:NEED TO DECIDE WHAT TO DO WITH MIN/MAX,RIGHT NOW by default min
            if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
                num = str(len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
                name_loss = name_loss + "_" + num
                c_name = c_name + "_" + num
            nodeReturn = handler_functions.handle_cross_entropy(self.node_map[e],name_loss,c_name)
            if nodeReturn != "":
                self.objectives(nodeReturn, e)
        elif self.node_map[e].get_op()=="Equal":
            nodeReturn= handler_functions.handle_accuracy(self.node_map[e])
            if nodeReturn!=None:
                self.data.evaluationResult.metric=nodeReturn
            else:
                print("LOGGING: Equal not metric.")
        elif "flatten" in e :
            self.handle_complex_layers(self.node_map[e],e,"flatten")
        elif "dense" in e and self.node_map[e].get_op() not in self.optimizer_operations:
            self.handle_complex_layers(self.node_map[e],e,"dense")
        elif "dropout" in e:
            self.handle_complex_layers(self.node_map[e],e,"dropout")
        elif "rnn" in e:
            self.handle_complex_layers(self.node_map[e],e,"rnn")
        elif "Placeholder"==self.node_map[e].get_op():
            has_dim=False
            for elem in self.node_map[e].get_output():
                for _ in elem.dim:
                    has_dim=True
                    break
            if has_dim==True:
                nodeReturn = handler_functions.handle_in_out_layer(self.node_map[e])
                self.insert_to_list(nodeReturn, e,"Layer")
        elif "mean_squared_error" in self.node_map[e].get_name()\
               and self.node_map[e].get_name().split("/")[0] not in self.discovered_loss:
            lname=self.node_map[e].get_name().split("/")[0]
            self.discovered_loss.append(self.node_map[e].get_name().split("/")[0])
            nodeReturn = handler_functions.handle_mean_square_error(self.node_map[e],self.current_network,lname)
            if nodeReturn!="":
                self.objectives(nodeReturn, e)
        elif self.node_map[e].get_op()=="Pow" or self.node_map[e].get_op()=="Square":
            #TODO:NEED TO DECIDE WHAT TO DO WITH MIN/MAX,RIGHT NOW by default min
            nodeReturn = handler_functions.handle_pow(self.node_map[e],self.current_network)
            if nodeReturn!="":
                self.objectives(nodeReturn, e)


    def find_in_out_layer(self):
        layers=self.data.annConfiguration.networks[self.current_network].layer.copy()
        layers_del=[]
        for layer in layers.keys():
            if self.data.annConfiguration.networks[self.current_network].layer[layer].previous_layer == [] or \
                    self.data.annConfiguration.networks[self.current_network].layer[layer].previous_layer == "":
                if self.data.annConfiguration.networks[self.current_network].layer[layer].next_layer == [] or \
                        self.data.annConfiguration.networks[self.current_network].layer[layer].next_layer == "":
                    output_node=handler_functions.handle_out_layer(self.data.annConfiguration.networks[self.current_network].layer[layer].node)
                    self.output_layer.append(output_node)
                    print("LOGGING:Found output layer ",output_node.name)
                    layers_del.append(layer)
                    self.data.annConfiguration.networks[self.current_network].output_layer.append(output_node)
                else:
                    for layer_in in self.data.annConfiguration.networks[self.current_network].layer.keys():
                        for i,input in enumerate(self.data.annConfiguration.networks[self.current_network].layer[layer_in].previous_layer):
                            #print(self.data.annConfiguration.networks[self.current_network].layer[layer_in].previous_layer)
                            if layer==input:
                                input_node = handler_functions.handle_in_layer(self.data.annConfiguration.networks[self.current_network].layer[layer].node)
                                self.input_layer.append(input_node)
                                print("LOGGING:Found input layer ", input_node.name)
                                del self.data.annConfiguration.networks[self.current_network].layer[layer_in].previous_layer[i]
                                self.input_layers.append(self.data.annConfiguration.networks[self.current_network].layer[layer_in])
                                layers_del.append(layer)
                                self.data.annConfiguration.networks[self.current_network].input_layer.append(input_node)
                                self.data.annConfiguration.networks[self.current_network].layer[
                                    layer_in].previous_layer.append(input_node.name)
                                break
        for elem in set(layers_del):
            del self.data.annConfiguration.networks[self.current_network].layer[elem]

    def prepare_strategy(self,batch,epochs,part_name):
        self.find_in_out_layer()
        tr_model=handler_functions.handle_trained_model(self.data.AnnConfig+"_trained_model")
        layers=self.data.annConfiguration.networks[self.current_network].layer
        counter=0
        for layer in layers.keys():
            if len(layers[layer].previous_layer)!=0:
                for elem_in in layers[layer].previous_layer:
                    counter+=1
                    weight=handler_functions.handle_weights('W'+str(counter),elem_in,layer)
                    tr_model.add_weight(weight)
        optimizer=self.data.annConfiguration.networks[self.current_network].optimizer
        optimizer_node=""
        for key in optimizer.keys():
            optimizer_node=optimizer[key]
        IOPipe=handler_functions.handle_dataset_pipe(self.current_network,self.input_layer,"train")
        tr_step = handler_functions.handle_training_single(part_name+"_training_step",self.current_network,IOPipe,optimizer_node,epochs,batch)
        tr_list_step=[]
        tr_list_step.append(tr_step)
        tr_session=handler_functions.handle_training_session(part_name+"_training_session",tr_list_step,"")
        tr_strategy=handler_functions.handle_training_strategy(part_name+"_training_strategy",tr_session,tr_model)
        self.data.evaluationResult.train_strategy=tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name]=tr_strategy

    def find_input_output(self):
        layers=self.annConfiguration.networks[self.current_network].layer
        for tmp in layers.keys():
            for prev in layers[tmp].previous_layer:
                if prev in self.layers().keys():
                    break
            self.annConfiguration.networks[self.current_network].input_layer=tmp
        for tmp in layers.keys():
            for next in layers[tmp].next_layer:
                if next in self.layers().keys():
                    break
            self.annConfiguration.networks[self.current_network].input_layer=tmp

    def check_multiple_networks(self):
        self.handle_possible_loss_functions()
        print("OBJECTIVES ARE :",self.data.annConfiguration.networks[self.current_network].objective.keys())
        if len(self.data.annConfiguration.networks[self.current_network].objective.keys())==1:
            print("LOGGING:Only one network presented,no more parsing.")
            return 0
        if len(self.data.annConfiguration.networks[self.current_network].objective.keys())==0:
            print("ERROR:There are no objective functions.Error occured.")
            return -1
        print("LOGGING:Multiple objective functions presented into ann configuration.")
        #TODO:check if the same objective have the same inputs,the one is metric the other one is loss func
        res=self.check_loss_metric()
        if res==True and len(self.data.annConfiguration.networks[self.current_network].objective.keys())==1:
            print("LOGGING:Only one network presented,no more parsing,multiple functions were evaluation and metric.")
            return 0
        net_cnt=0
        network_outputs=[]
        objectives=[]
        for l in self.data.annConfiguration.networks[self.current_network].objective.keys():
            objectives.append(l)
        #self.find_combined_losses(objectives)
        for obj in sorted(objectives):
            print("Encountered objective ",self.data.annConfiguration.networks[self.current_network].objective[obj].name)
            obj_func=self.data.annConfiguration.networks[self.current_network].objective[obj]
            (output,net_cnt)=self.find_network(obj_func,network_outputs,self.current_network,net_cnt)
            for name in output:
                network_outputs.append(name)

        self.insert_training()
        del self.data.annConfiguration.networks[self.current_network]
        return 1

    def handle_possible_loss_functions(self):
        for poss in self.possible_loss_function.keys():
            cnt=0
            print(self.possible_loss_function[poss])
            for layer in self.data.annConfiguration.networks[self.current_network].layer:
                if len(set(self.possible_loss_function[poss]).intersection(set(self.data.annConfiguration.networks[self.current_network].layer[layer].output_nodes)))>0:
                        cnt+=1
                elif self.data.annConfiguration.networks[self.current_network].layer[layer].activation!=None:
                    if len(set(self.possible_loss_function[poss]).intersection(set([self.data.annConfiguration.networks[self.current_network].layer[layer].activation.name]))) > 0:
                        cnt+=1
            if cnt==2:
                name = "objective_function"
                if len(self.data.annConfiguration.networks[self.current_network].objective.keys()) != 0:
                    name = name + "_" + str(
                        len(self.data.annConfiguration.networks[self.current_network].objective.keys()))
                nodeReturn = handler_functions.handle_mean_square_error(self.node_map[poss],self.current_network,poss)
                if nodeReturn != "":
                    nodeReturn = handler_functions.handle_objective(name, nodeReturn)
                    if nodeReturn != "":
                        self.insert_to_list(nodeReturn, name, "Objective")


    def check_loss_metric(self):
        #TODO:If found two loss functions with same input ,one of them
        for obj in self.data.annConfiguration.networks[self.current_network].objective.keys():
            inputs=self.data.annConfiguration.networks[self.current_network].objective[obj].cost_function.loss.node.get_inputs()
            inputs=[x.get_name() for x in inputs]
            for obj_in in self.data.annConfiguration.networks[self.current_network].objective.keys():
                if obj_in!=obj:
                    inputs_in=[]
                    inputs_=self.data.annConfiguration.networks[self.current_network].objective[obj_in].cost_function.loss.node.get_inputs()
                    for input_ in inputs_:
                        inputs_in.append(input_.get_name())
                    if len(set(inputs_in).intersection(set(inputs))) > 0:
                        print("LOGGING:Found 2 loss function with same input.One of them is evaluation metric.")
                        type=self.data.annConfiguration.networks[self.current_network].objective[obj_in].cost_function.loss.type
                        node=self.data.annConfiguration.networks[self.current_network].objective[obj_in].cost_function.loss.node
                        if type=="MSE":
                            nodeReturn = handler_functions.handle_mse_metric(node)
                            self.data.evaluationResult.metric = nodeReturn
                        else:
                            print("ERROR:Cannot find type of loss function to make it evaluation")
                            import sys
                            sys.exit()
                        del self.data.annConfiguration.networks[self.current_network].objective[obj_in]
                        return True
        return False


    def find_network(self,obj_func,net_outputs,network,counter):
        loss=obj_func.cost_function.loss
        if loss.input_nodes!=[]:
            children=loss.input_nodes
        else:
            node_loss = self.node_map[loss.node.get_name()]
            children = node_loss.get_inputs()
        print("\n\n\n\n\nAbout to begin searching from ",set([x.get_name() for x in children]))
        outputs=self.find_network_output_layer(children)
        layers = {}
        inputs=[]
        for name in set(outputs):
            (tinput,tlayers)=self.find_network_input_layer_and_layers(name,net_outputs)
            for inp in tinput:
                if inp not in inputs:
                    inputs.append(inp)
            for k,v in tlayers.items():
                layers[k]=v
        self.insert_network(layers, network+"_"+str(counter), inputs, outputs, obj_func)
        st=""
        for x in layers.keys():
            st=st+"\n"+x
        print("\n\n\n\n\n\n\n\n\n\nLayers=",st," \n-------------------------------\nLoss:",loss.node.get_name())
        print("--------------------\nInput=",inputs)
        print("Net_Outputs=", set(outputs))
        counter+=1
        return (outputs,counter)

    def find_network_input_layer_and_layers(self,output,net_outputs):
        layers = self.data.annConfiguration.networks[self.current_network].layer
        prev=output
        print("Start searching from output ",output)
        if prev not in layers.keys():
            print("Previous layer not in keys ",prev)
            prev=self.nodes_to_layers[prev]
            print("Previous translated to ",prev)
        prev=layers[prev].previous_layer
        net_layers={}
        ret_layer=[]
        while prev!=[]:
            temp=[]
            for prev_layer in prev:
                if layers[prev_layer].__class__.__name__ == "InputLayer" \
                        or layers[prev_layer].__class__.__name__ == "OutputLayer" \
                        or layers[prev_layer].previous_layer==""\
                        or layers[prev_layer].previous_layer==[] \
                        or layers[prev_layer].name in net_outputs:
                            print("found input=",prev_layer)
                            print("class=",layers[prev_layer].__class__.__name__ )
                            print("previous=", layers[prev_layer].previous_layer)
                            print("RETURNING LAYERS =",layers)
                            ret_layer.append(prev_layer)
                else:
                    net_layers[prev_layer]=layers[prev_layer]
                    for elem in layers[prev_layer].previous_layer:
                        temp.append(elem)
            prev=temp

        return (ret_layer, net_layers)

    def find_network_output_layer(self,children):
        print("LOGGING:Start searching for output node of ",[x.get_name() for x in children])
        outputs=[]
        while children!=[]:
            tmp=[]
            for elem in children:
                print("Children node ",elem.get_name())
                if elem.get_op() != "Placeholder":
                    if elem.get_name() in self.data.annConfiguration.networks[self.current_network].layer.keys():
                        outputs.append(elem.get_name())
                        print("LOGGING:Found output layer = ",elem.get_name())
                    elif elem.get_op() in self.intermediate_operations or elem.get_op() in self.activation_operations or elem.get_op()=="Add"\
                           or elem.get_op()=="Log":
                        for input in elem.get_inputs():
                            if input not in tmp:
                                tmp.append(input)
                    else:
                        print("LOGGING:Encountered intermediate node ",elem.get_name(),".Children added.")
                        for input in elem.get_inputs():
                            tmp.append(input)

            children=tmp
        print("LOGGING:Find the following output layers:",outputs)
        return outputs


    def insert_network(self,layers,name,inputs,outputs,objective_func):
        self.data.init_new_network(name)
        for layer in layers.keys():
            self.data.annConfiguration.networks[name].layer[layer]=layers[layer]
        for input in inputs:
            node=self.data.annConfiguration.networks[self.current_network].layer[input].node
            input_layer=handler_functions.handle_in_layer(node)
            self.data.annConfiguration.networks[name].input_layer.append(input_layer)
        for output in outputs:
            node = self.data.annConfiguration.networks[self.current_network].layer[output].node
            output_layer = handler_functions.handle_out_layer(node)
            self.data.annConfiguration.networks[self.current_network].output_layer.append(output_layer)
        if objective_func!="":
            self.data.annConfiguration.networks[name].objective[objective_func.name]=objective_func

    def insert_training(self):
        trSteps=[]
        networks=self.data.annConfiguration.networks.keys()
        for network in networks:
            if self.current_network!=network:
                layers=[]
                for layer in self.data.annConfiguration.networks[network].layer.keys():
                    layers.append(layer)
                #print("\n\n\nStart searching for optimizer for ",layers,"\n\n\n")
                optimizer=self.find_optimizer(layers)
                if optimizer!=None:
                    self.data.annConfiguration.networks[network].optimizer[optimizer.name]=optimizer
                    input_layer=self.data.annConfiguration.networks[network].input_layer
                    print("LOGGING:Input_layer ",[x.name for x in input_layer]," Optimizer ",optimizer.name)
                    IOPipe = handler_functions.handle_dataset_pipe(network, input_layer, "train")
                    #TODO:EDW TI NA KANW POU XRIAZONTE POLLAPLA EPOCH AND BATCHES???????
                    tr_step = handler_functions.handle_training_single(network + "_training_step", network, IOPipe,optimizer, 0, 0)
                    trSteps.append(tr_step)
                else:
                    st=""
                    for x in layers:
                        st=st+"\n"+x
                    print("ERROR:The following layers are not part of an optimization procedure:",st)
        tr_model = handler_functions.handle_trained_model(self.data.AnnConfig + "_trained_model")
        layers = self.data.annConfiguration.networks[self.current_network].layer
        counter = 0
        for layer in layers.keys():
            if len(layers[layer].previous_layer) != 0:
                for elem_in in layers[layer].previous_layer:
                    counter += 1
                    weight = handler_functions.handle_weights('W' + str(counter), elem_in, layer)
                    tr_model.add_weight(weight)
        tr_session = handler_functions.handle_training_session(self.current_network + "_training_session", trSteps, "")
        tr_strategy = handler_functions.handle_training_strategy(self.current_network + "_training_strategy", tr_session, tr_model)

        if self.data.evaluationResult.metric!="":
            IOPipe = handler_functions.handle_dataset_pipe(self.current_network, self.output_layer, "test")
            self.data.evaluationResult.IOPipe = IOPipe
        self.data.evaluationResult.train_strategy = tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name] = tr_strategy
        self.data.evaluationResult.ann_conf = self.data.annConfiguration

    def find_optimizer(self,layers):
        for optimizer in self.data.annConfiguration.networks[self.current_network].optimizer.keys():
            found_gradient = False
            gradient=""
            print("Start searching for ",optimizer)
            for name in self.node_map.keys():
                if name.startswith(optimizer+"/"):
                    for input in self.node_map[name].get_inputs():
                        #Try to match the optimizer with input that contain gradient word.This will lead us to a layer that we can afterwards
                        #assume that the optimizer is connected with the respective network.
                        if input.get_name().startswith("gradient"):
                            print(name,"---",input.get_name())
                            print("LOGGING:Successfuly matched optimizer to gradient. Information:\nOptimizer=",optimizer,"\nGradient=",input.get_name().split("/")[0])
                            gradient=input.get_name().split("/")[0]
                            found_gradient=True
                            break
                if found_gradient==True:
                    break
            node_layers=[]
            for layer in layers:
                #Attempt to find all layers of a network and add their respective names into one list to form the
                #node name that contains gradient+layer name
                if layer not in self.data.annConfiguration.networks[self.current_network].layer.keys():
                    for elem in self.layers_to_nodes[layer]:
                        node_layers.append(elem)
                else:
                    node_layers.append(layer)
            grad_layers_names=[x+"_grad" for x in node_layers]
            print("gradient list layer =",grad_layers_names)
            for name in self.node_map.keys():
                name_list=name.split("/")
                found=False
                for name_l in name_list:
                    for elem in grad_layers_names:
                        elem_list=[]
                        if "/" in elem:
                            elem_list=elem.split("/")
                        else:
                            elem_list=elem
                        elems_in_both_lists = set(name_list) & set(elem_list)
                        if len(elems_in_both_lists)==len(elem_list):
                            found=True
                            break

                    if (found==True) or (name_l in grad_layers_names and gradient==name.split("/")[0]):
                        print("FOUND:\nOptimizer=",optimizer,"\nLname=",grad_layers_names,"\ngradient=",gradient)
                        print(self.data.annConfiguration.networks[self.current_network].optimizer[optimizer])
                        return self.data.annConfiguration.networks[self.current_network].optimizer[optimizer]


    def clear_layers(self):
        layer_names=list(self.data.annConfiguration.networks[self.current_network].layer.keys())
        while layer_names!=[]:
            for layer in self.data.annConfiguration.networks[self.current_network].layer.keys():
                inner=self.data.annConfiguration.networks[self.current_network].layer[layer].inner_nodes
                if [ layer_names[0] for x in inner if layer_names[0]==x]!=[]:
                    del self.data.annConfiguration.networks[self.current_network].layer[layer_names[0]]
                    print("LOGGING:Deleted inner node ",layer_names[0])
                    break
            layer_names.remove(layer_names[0])
        objs = list(self.data.annConfiguration.networks[self.current_network].objective)
        layer_names = list(self.data.annConfiguration.networks[self.current_network].layer.keys())
        #print("LOGGING:Starting loss search for inner nodes")
        for obj in objs:
            loss=self.data.annConfiguration.networks[self.current_network].objective[obj].cost_function.loss
            while layer_names!=[]:
                #print("LOGGING:Starting loss search for inner nodes=",loss.inner_nodes)
                if [ layer_names[0] for x in set(loss.inner_nodes) if layer_names[0]==x]!=[]:
                    del self.data.annConfiguration.networks[self.current_network].layer[layer_names[0]]
                layer_names.remove(layer_names[0])
                print(len(layer_names))


    '''
        def handle_autoencoder(self,num_of_layers):
            prev_val=""
            input_layer=""
            output_layer=""
            network_e=self.current_network+'_ENCODER'
            network_e_list=[]
            network_d = self.current_network + '_DECODER'
            network_d_list=[]
            found_output=False
            for layers in num_of_layers:
                print(layers)
                for layer in layers:
                    print("LAYER=",layer)
                    val=layer[1]
                    if prev_val=="":
                        input_layer=layer[0]
                    elif prev_val>val:
                        network_e_list.append(layer[0])
                    else:
                        if found_output==False and len(network_d_list)==0:
                            found_output=True
                            output_layer=network_e_list[-1]
                            network_d_list.append(layer[0])
                            del network_e_list[-1]
                        else:
                            network_d_list.append(layer[0])
                    prev_val=val
            output_layer_d = network_d_list[-1]
            del network_d_list[-1]
            print("ENCODER=",network_e_list)
            print("DECODER=", network_d_list)
            print("INPUT=", input_layer)
            print("OUTPUT=", output_layer)
            print("OUTPUT_D=", output_layer_d)
            if len(self.data.annConfiguration.networks[self.current_network].objective.keys())==1:
                obj_name=list(self.data.annConfiguration.networks[self.current_network].objective.keys())[0]
                obj_func=self.data.annConfiguration.networks[self.current_network].objective[obj_name]
                optimizer = self.data.annConfiguration.networks[self.current_network].optimizer
                if len(optimizer.keys()) > 1:
                    print("SHOULD NOT BE MORE THAN 1")
                    print(optimizer)
                if len(optimizer.keys()) == 0:
                    print("SHOULD NOT BE MORE THAN 0")
                optimizer_node = ""
                self.insert_network(network_e_list,network_e,input_layer,output_layer,"")
                self.insert_network(network_d_list, network_d,output_layer, output_layer_d,obj_func)
                IOPipe = handler_functions.handle_dataset_pipe(network_e, input_layer, "train")
                tr_step = handler_functions.handle_training_single(network_e + "_training_step", network_e,
                                                                   IOPipe, optimizer_node, epochs, batch)
                tr_session = handler_functions.handle_training_session(part_name + "_training_session", tr_step, "")
                tr_strategy = handler_functions.handle_training_strategy(part_name + "_training_strategy", tr_session,
                                                                         tr_model)
                self.data.evaluationResult.train_strategy = tr_strategy
                self.data.annConfiguration.training_strategy[tr_strategy.name] = tr_strategy
            else:
                import sys
                for elem in self.data.annConfiguration.networks[self.current_network].objective.keys():
                    print("OBJ=",elem.name)
                print("Multiple Objective functions.")
                sys.exit()

        def check_if_ae(self):
            if len(self.input_layer)!=1:
                print("ERROR:Input is different to 1")
                return
            num_of_layers=[]
            layer_num=0
            list_of_layers=set(self.input_layers)
            tmp_ls=[]
            tmp = []
            tmp.append(self.input_layer[0].name)
            tmp.append(self.input_layer[0].num_layer)
            tmp_ls.append(tmp)
            num_of_layers.append(tmp_ls)
            while len(list_of_layers)!=0:
                layer_num += 1
                temp=[]
                tmp_ls = []
                for l in list_of_layers:
                    print("PSARRAS:LOL-L",l.name)
                    tmp=[]
                    tmp.append(l.name)
                    tmp.append(l.num_layer)
                    tmp_ls.append(tmp)
                    for nl in l.next_layer:
                        print("PSARRAS:NL in LOL-L",nl)
                        print("LAYER:",l.name,"->NEXT LAYER:",nl)
                        layer=self.data.annConfiguration.networks[self.current_network].layer[nl]
                        temp.append(layer)
                if tmp_ls!=[]:
                    num_of_layers.append(tmp_ls)
                list_of_layers=temp

            print(num_of_layers)
            print(layer_num)
            print(num_of_layers[0][0][1])
            print(num_of_layers[layer_num-1][0][1])
            if num_of_layers==[] or len(num_of_layers[0])>1 or len(num_of_layers[layer_num-1])>1:
                print("ERROR:DOUBLE INPUT OR OUTPUT,OR NO NUM_OF_LAYERS")
                #import sys
                #sys.exit()
            elif layer_num>1 and layer_num%2==0 and num_of_layers[0][0][1]==num_of_layers[-1][0][1]:
                print("mpika=",num_of_layers)
                self.handle_autoencoder(num_of_layers)
                return
            print("There is no auto encoder as input.")
        '''