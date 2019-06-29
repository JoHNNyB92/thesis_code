from virtuosoWrapper.annet_o_data_management import annet_o_data_management as dataMgnt
from NetworkEvaluation.evaluation_result import evaluation_result
from nodes import handler_functions
from nodes.nodeEntity import nodeEntity as nodeClass
from Network.network import network
from copy import copy

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
        #To give different names to layers that exist in multiple networks
        self.sameLayerCounter=0
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
    def insert_to_evaluation_pipe(self,network):
        IOPipe=handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[network],"test")
        self.check_metric(IOPipe, network)
        #self.data.evaluationResult.IOPipe=IOPipe
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
                nodeReturn = handler_functions.handle_rms_prop(self.node_map[e], real_name)
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

    '''
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
    '''
    def prepare_strategy(self,batch,epochs,part_name):
        #self.find_in_out_layer()
        for obj in self.data.annConfiguration.networks[self.current_network].objective.keys():
            (n_name,_,_,_)=self.find_network(self.data.annConfiguration.networks[self.current_network].objective[obj],[],self.current_network,0)
        tr_model=handler_functions.handle_trained_model(self.data.AnnConfig+"_trained_model")
        layers=self.data.annConfiguration.networks[n_name].layer
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
        IOPipe=handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[n_name],"train")
        tr_step = handler_functions.handle_training_single(part_name+"_training_step",n_name,IOPipe,optimizer_node,epochs,batch)
        tr_list_step=[]
        tr_list_step.append(tr_step)
        tr_session=handler_functions.handle_training_session(part_name+"_training_session",tr_list_step,"")
        tr_strategy=handler_functions.handle_training_strategy(part_name+"_training_strategy",tr_session,tr_model)
        self.data.evaluationResult.train_strategy=tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name]=tr_strategy
        self.insert_to_evaluation_pipe(n_name)
        del self.data.annConfiguration.networks[self.current_network]
    '''
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
    '''
    def check_multiple_networks(self):
        self.handle_possible_loss_functions()
        print("OBJECTIVES ARE :",self.data.annConfiguration.networks[self.current_network].objective.keys())
        if len(self.data.annConfiguration.networks[self.current_network].objective.keys())==1:
           #self.transform_layers_to_ins_outs(self.current_network,self.data.annConfiguration.networks[self.current_network].keys())
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
        optimizer_per_layer = self.find_optimizer()
        #self.find_combined_losses(objectives)
        for obj in sorted(objectives):
            print("Encountered objective ",self.data.annConfiguration.networks[self.current_network].objective[obj].name)
            obj_func=self.data.annConfiguration.networks[self.current_network].objective[obj]
            (output_layers,output,net_cnt,optimizer)=self.find_network(obj_func,network_outputs,self.current_network,net_cnt,optimizer_per_layer)
            for name in output:
                network_outputs.append(name)

        self.insert_training(optimizer)
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


    def find_network(self,obj_func,net_outputs,network,counter,opl):
        loss=obj_func.cost_function.loss
        if loss.input_nodes!=[]:
            children=loss.input_nodes
        else:
            node_loss = self.node_map[loss.node.get_name()]
            children = node_loss.get_inputs()
        print("LOGGING:About to begin searching from ",set([x.get_name() for x in children]))
        #Returns List of node outputs,list of layer outputs,a dictionary of (layer name)-(output placeholder) , if exists
        (outputs,layer_outputs,datasets)=self.find_network_output_layer(children)
        layers = {}
        inputs=[]
        for ind,mem in enumerate(layer_outputs):
            (tinput,tlayers)=self.find_network_input_layer_and_layers(mem,net_outputs)
            for inp in tinput:
                if inp not in inputs:
                    inputs.append(inp)
            for k,v in tlayers.items():
                layers[k]=v
        l_outputs=[]


        for output in outputs:
            l_outputs.append(self.data.annConfiguration.networks[self.current_network].layer[output.get_name()])
        (optimizer,layers,inputs,counter)=self.handle_optimizer_per_layer(layers,opl,inputs,network,counter)
        n_name = network + "_" + str(counter)
        self.insert_network(layers,n_name, inputs, l_outputs, obj_func,datasets)
        counter += 1
        st=""
        for x in layers.keys():
            st=st+"\n"+x
        return (n_name,outputs,counter,optimizer)


    def find_network_input_layer_and_layers(self,output,net_outputs):
        layers = self.data.annConfiguration.networks[self.current_network].layer
        prev=output.name
        net_layers = {}
        net_layers[prev] = output
        print("Start searching from output ",output.name)
        if prev not in layers.keys():
            #print("Previous layer not in keys ",prev)
            prev=self.nodes_to_layers[prev]
            #print("Previous translated to ",prev)
        prev=layers[prev].previous_layer
        ret_layer = []
        while prev!=[]:
            temp=[]
            for prev_layer in prev:
                if layers[prev_layer].is_input==True:
                    #print("LOGGING:Discovered input layer = ",prev_layer)
                    ret_layer.append(layers[prev_layer].placeholder)
                    net_layers[layers[prev_layer].placeholder]=layers[layers[prev_layer].placeholder]
                    net_layers[prev_layer] = layers[prev_layer]
                else:
                    #print("LOGGING:Discovered intermediate layer = ",prev_layer)
                    net_layers[prev_layer]=layers[prev_layer]
                    for elem in layers[prev_layer].previous_layer:
                        temp.append(elem)
            prev=list(set(temp))
        return (ret_layer, net_layers)

    def find_network_output_layer(self,children):
        print("LOGGING:Start searching for output node of ",[x.get_name() for x in children])
        outputs=[]
        layer_outputs=[]
        datasets={}
        found_placeholder=""
        while list(set(children))!=[]:
            tmp=[]
            for elem in list(set(children)):
                #print("Children node ",elem.get_name())
                if elem.get_op() != "Placeholder":
                    if elem.get_name() in self.data.annConfiguration.networks[self.current_network].layer.keys():
                        layer_outputs.append(self.data.annConfiguration.networks[self.current_network].layer[elem.get_name()])
                        if found_placeholder!="":
                            datasets[elem.get_name()] = found_placeholder
                        outputs.append(elem)
                        #print("LOGGING:Found output layer = ", elem.get_name())
                    elif elem.get_op() in self.intermediate_operations or elem.get_op() in self.activation_operations or elem.get_op()=="Add"\
                           or elem.get_op()=="Log":
                        for input in elem.get_inputs():
                            if input not in tmp:
                                tmp.append(input)
                    else:
                        #print("LOGGING:Encountered intermediate node ",elem.get_name(),".Children added.")
                        for input in elem.get_inputs():
                            #print("O:",elem.get_name()," NIN:",input.get_name())
                            tmp.append(input)
                else:
                    if len(outputs)!=0:
                        datasets[outputs[-1].get_name()]=elem
                    found_placeholder=elem
            children=list(set(tmp))
        print("LOGGING:Find the following output layers:",[x.get_name() for x in outputs])
        return (outputs,layer_outputs,datasets)

    def handle_optimizer_per_layer(self,layers, opl,inputs,network_,counter):
        sum={}
        all_opt_layers=[]
        new_names={}
        #Identify and put into a list all discovered layers whose train belongs to an optimizer.
        for key in opl.keys():
            for l in opl[key]:
                all_opt_layers.append(l)
        #Identify which layers do not have an optimizer attached(e.g. concatLayer)
        no_opt_layers=self.data.annConfiguration.networks[self.current_network].layer.keys()-all_opt_layers

        #Copy the layers into a temporary object and remove from it the layers not belonging to an optimizer.
        t_layers=layers.copy()
        for layer in layers:
            if layer in no_opt_layers:
                del t_layers[layer]

        #Dictionary to hold layer-optimizer pair.
        temp_layer_optimizer = {}
        for key in opl.keys():
            #Sum contains 1 on whether a layer of the network belongs to this optimizer or 0 if it is part of another one.
            sum[key]=[]
            non_optimizer={}
            #Filling sum with 1/0 if it belongs or not to optimizer 'key',also add layer optimizer to temp_layer_optimizer
            for layer in t_layers:
                if layer in opl[key]:
                    sum[key].append(1)
                    if layer in temp_layer_optimizer.keys():
                        temp_layer_optimizer[layer].append(key)
                    else:
                        temp_layer_optimizer[layer]=key
                else:
                    non_optimizer[layer]=key
                    sum[key].append(0)
            #If the sum of the 1 is the same as the networks layers length that means that all the
            #layers belong to one optimizer,thus to the same network.
            if sum[key].count(1)==len(t_layers.keys()):
                print("LOGGING:All layers have the same network=",layers)
                return (key,layers,inputs,counter)
            #We assume here that a network is considered to have a specific optimizer,if two are present,into
            #the one having the most layers.The other one,is considered a new network.
            if sum[key].count(1)>=(len(t_layers)/2):
                new_networks={}
                inputs_list=[]
                tt_layers=t_layers.copy()
                for layer in t_layers:
                    #If layer does not belong to this optimizer
                    if layer in non_optimizer.keys():
                        #Insert it into a new list,with layers probable to be input to the bigger network,
                        #later there will be checks on whether this is input or part of another network.
                        inputs_list.append(layer)
                        for pl in layers[layer].previous_layer:
                            if pl in inputs:
                                #New network presented,we are goind to remove the input of the current networ
                                #and insert it as input to the new network.
                                if non_optimizer[layer] not in new_networks.keys():
                                    name=non_optimizer[layer]
                                    new_networks[non_optimizer[layer]]=network(name)
                                new_networks[non_optimizer[layer]].layer[pl+"--"+str(self.sameLayerCounter)]=copy(layers[pl])
                                new_names[pl]=pl+"--"+str(self.sameLayerCounter)
                                new_networks[non_optimizer[layer]].layer[pl +"--" + str(self.sameLayerCounter)].name=pl + "--" + str(self.sameLayerCounter)
                                new_networks[non_optimizer[layer]].layer[pl + "--" + str(self.sameLayerCounter)].sameLayer = pl
                                new_networks[non_optimizer[layer]].input_layer.append(pl+"--"+str(self.sameLayerCounter))
                                self.sameLayerCounter+=1
                                inputs.remove(pl)
                        del tt_layers[layer]
                new_inputs = []
                for input in inputs_list:
                    next_layer=layers[input].next_layer
                    found=False
                    print("Input=",input)
                    for nl in next_layer:
                        #From the layers belonging to another network we check whether the next layer is part of the current network
                        #If it is,then this layer is output of the new network and input of new one
                        if nl in tt_layers:
                            print("1YEAH FINAL INPUT=",input)
                            new_inputs.append(input)
                            new_networks[non_optimizer[input]].layer[input+"--"+str(self.sameLayerCounter)] = copy(layers[input])
                            new_names[input] = input+"--"+str(self.sameLayerCounter)
                            new_networks[non_optimizer[input]].layer[
                                input + "--" + str(self.sameLayerCounter)].name = input + "--" + str(self.sameLayerCounter)

                            new_networks[non_optimizer[input]].layer[input +"--"+str(self.sameLayerCounter)].sameLayer=input
                            new_networks[non_optimizer[input]].output_layer.append(new_networks[non_optimizer[input]].layer[input+"--"+str(self.sameLayerCounter)])
                            self.sameLayerCounter+=1
                            found=True
                            break
                        elif nl in no_opt_layers:
                            #There is a change next layer to does not have an optimizer,thus we move on to check the next of it,
                            #until we find a layer belonging to current network
                            done = False
                            next_l=[]
                            next_l.append(nl)
                            while done==False:
                                tmp=[]
                                for l in next_l:
                                    next_layer = layers[l]
                                    for elem in next_layer:
                                        if elem in layers:
                                            if elem in tt_layers.keys():
                                                print("2YEAH FINAL INPUT=", input)
                                                new_inputs.append(input)
                                                new_networks[non_optimizer[input]].layer[
                                                    input + "--" + str(self.sameLayerCounter)] = copy(layers[input])
                                                new_networks[non_optimizer[input]].layer[
                                                    input + +"--" + str(self.sameLayerCounter)].sameLayer = input
                                                new_networks[non_optimizer[input]].output_layer.append(
                                                    new_networks[non_optimizer[input]].layer[
                                                        input + "--" + str(self.sameLayerCounter)])
                                                new_names[input] = input + "--" + str(self.sameLayerCounter)
                                                new_networks[non_optimizer[input]].layer[
                                                    input + "--" + str(self.sameLayerCounter)].name = input + "--" + str(
                                                    self.sameLayerCounter)
                                                self.sameLayerCounter+=1
                                                done=True
                                                found=True
                                                break
                                        else:
                                            if elem in no_opt_layers:
                                                tmp.append(elem)
                    if found==False:
                        #If nothing was found( a layer to be input to the current network)
                        #we add this layer to the new network as simple intermediate layer.
                        new_networks[non_optimizer[input]].layer[input+"--"+str(self.sameLayerCounter)]=copy(layers[input])
                        new_names[input] = input + "--" + str(self.sameLayerCounter)
                        new_networks[non_optimizer[input]].layer[
                            input + "--" + str(self.sameLayerCounter)].name = input + "--" + str(self.sameLayerCounter)
                        new_networks[non_optimizer[input]].layer[input + "--" + str(self.sameLayerCounter)].sameLayer=input
                        self.sameLayerCounter+=1
                        del layers[input]
                for x in inputs:
                    new_inputs.append(x)
                for key in opl.keys():
                    if key in new_networks.keys():
                        name=network_ + "_" + str(counter)
                        counter+=1
                        for layer in new_networks[key].layer.keys():
                            for nl in new_networks[key].layer[layer].next_layer:
                                if nl in new_names.keys():
                                    new_networks[key].layer[layer].next_layer.remove(nl)
                                    new_networks[key].layer[layer].next_layer.append(new_names[nl])
                            for pl in new_networks[key].layer[layer].previous_layer:
                                if pl in new_names.keys():
                                    new_networks[key].layer[layer].previous_layer.remove(pl)
                                    new_networks[key].layer[layer].previous_layer.append(new_names[pl])

                        self.insert_network(new_networks[key].layer,name,
                                            new_networks[key].input_layer,
                                            new_networks[key].output_layer,"",{})
                print("LOGGING:Network changed-Layers:", layers)
                print("LOGGING:Network changed-Key:", key)
                print("LOGGING:Network changed-New Inputs:", new_inputs)
                return (key,layers,new_inputs,counter)

    def insert_network(self,layers,name,inputs,outputs,objective_func,datasets):
        self.data.init_new_network(name)
        print("\n---------------START NEW NETWORK-----------------\n")
        for input in inputs:
            input_layer=handler_functions.handle_in_layer(layers[input])
            input_layer.placeholder=layers[input].placeholder
            print("LOGGING:["+name+"]Input is =", input," Dataset is = ",input_layer.placeholder)
            self.data.annConfiguration.networks[name].input_layer.append(input_layer)
            #del self.data.annConfiguration.networks[self.current_network].layer[input]
            del layers[input]
        for output in outputs:
            #node = output.node
            print("LOGGING:["+name+"]Output is = ",output.name)
            output_layer = handler_functions.handle_out_layer(output)
            self.data.annConfiguration.networks[name].output_layer.append(output_layer)
            #del self.data.annConfiguration.networks[self.current_network].layer[output.name]
            #del layers[output.name]
        for layer in layers.keys():
            print("LOGGING:["+name+"]Layer is ",layer)
            self.data.annConfiguration.networks[name].layer[layer]=layers[layer]
        if objective_func!="":
            print("LOGGING:["+name+"]Objective func is = ",objective_func.name)
            self.data.annConfiguration.networks[name].objective[objective_func.name]=objective_func
        if datasets.keys()!=[]:
            print("LOGGING:["+name+"]Dataset is = ", datasets)
            self.data.annConfiguration.networks[name].datasets=datasets
        print("\n---------------END NEW NETWORK-----------------\n")

    def insert_training(self,optimizer):
        trSteps=[]
        optimizer=self.data.annConfiguration.networks[self.current_network].optimizer[optimizer]
        networks=self.data.annConfiguration.networks.keys()
        for network in networks:
            if self.current_network!=network:
                layers=[]
                for layer in self.data.annConfiguration.networks[network].layer.keys():
                    layers.append(layer)
                    self.data.annConfiguration.networks[network].optimizer[optimizer]=optimizer
                    input_layer=self.data.annConfiguration.networks[network].input_layer
                    print("LOGGING:Input_layer ",[x.name for x in input_layer]," Optimizer ",optimizer.name)
                    IOPipe = handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[network],"train")
                    #TODO:EDW TI NA KANW POU XRIAZONTE POLLAPLA EPOCH AND BATCHES???????
                    tr_step = handler_functions.handle_training_single(network + "_training_step", network, IOPipe,optimizer, 0, 0)
                    trSteps.append(tr_step)
                else:
                    st=""
                    for x in layers:
                        st=st+"\n"+x
                    print("ERROR:The following layers are not part of an optimization procedure:",st)
        tr_model = handler_functions.handle_trained_model(self.data.AnnConfig + "_trained_model")
        layers = self.data.annConfiguration.networks[network].layer
        counter = 0
        for layer in layers.keys():
            if len(layers[layer].previous_layer) != 0:
                for elem_in in layers[layer].previous_layer:
                    counter += 1
                    weight = handler_functions.handle_weights('W' + str(counter), elem_in, layer)
                    tr_model.add_weight(weight)
        tr_session = handler_functions.handle_training_session(network + "_training_session", trSteps, "",)
        tr_strategy = handler_functions.handle_training_strategy(network + "_training_strategy", tr_session, tr_model)

        IOPipe = handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[network],"test")
        self.check_metric(IOPipe,network)
        self.data.evaluationResult.train_strategy = tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name] = tr_strategy
        self.data.evaluationResult.ann_conf = self.data.annConfiguration

    def check_metric(self,IOPipe,network):
        if self.data.evaluationResult.metric!="":
            res=self.check_evaluation(IOPipe)
            print("res=",res)
            if res==True:
                self.data.evaluationResult.IOPipe = IOPipe
            else:
                for elem in IOPipe:
                    self.data.annConfiguration.networks[network].hang_dp.append(elem)
        else:
            for elem in IOPipe:
                self.data.annConfiguration.networks[network].hang_dp.append(elem)
        self.data.evaluationResult.ann_conf = self.data.annConfiguration

    def check_evaluation(self,IOPipes):
        for IOPipe in IOPipes:
            output=IOPipe.dataset.name
            node=self.node_map[self.data.evaluationResult.metric.name]
            children = node.get_inputs()
            while children != []:
                tmp = []
                for elem in children:
                    if self.node_map[elem.get_name()].get_op() == "Placeholder":
                        if self.node_map[elem.get_name()].get_name()==output:
                            print("LOGGING:Associated metric ",self.data.evaluationResult.metric.name," with dataset ",output," .")
                            return True
                        else:
                            #If it encounters a different placeholder,that means that the one searching is wrong.
                            return False
                    else:
                        for x in self.node_map[elem.get_name()].get_inputs():
                            tmp.append(x)
                children = tmp
        return False

    def find_optimizer(self):
        layers=self.data.annConfiguration.networks[self.current_network].layer
        optimizer_per_layer = {}
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
            grad_layers_names=[gradient+"/"+x+"_grad" for x in node_layers]
            #Flatter for exampel requires different handling.
            shape_grad=[gradient+"/"+x+"/Reshape_grad/Shape" for x in node_layers]
            grad_layers_names=shape_grad+grad_layers_names
            print("gradient list layer =",grad_layers_names)
            for name in self.node_map.keys():
                name_list=name.split("/")
                found=False
                elem_=""
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
                            elem_ = elem.replace("/Reshape_grad/Shape", "")
                            elem_=elem_.replace("_grad","")
                            elem_ = elem_.replace(gradient+"/", "")
                            break

                    if (found==True) or (name_l in grad_layers_names and gradient==name.split("/")[0]):
                        if optimizer not in optimizer_per_layer.keys():
                            optimizer_per_layer[optimizer]=[]
                        if elem_ not in optimizer_per_layer[optimizer]:
                            optimizer_per_layer[optimizer].append(elem_)
                            print("FOUND:\nOptimizer=",optimizer,"\nLname=",elem_,"\ngradient=",gradient)
                            print(optimizer_per_layer)
        print("LOCO=",optimizer_per_layer)
        for key in optimizer_per_layer.keys():
            str=""
            for x in optimizer_per_layer[key]:
                str=str+x+"\n"
            print("Key:",key,"List:",str)
        return (optimizer_per_layer)


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

    def transform_layers_to_ins_outs(self,inputs,curr):
        for elem in self.data.annConfiguration.networks[self.current_network].layer.keys():
            # If only one input and that input is placeholder then it an input layer
            if nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].is_input == True and \
                    len(nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[
                            elem].previous_layer) == 1:
                inputs.append(elem)

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