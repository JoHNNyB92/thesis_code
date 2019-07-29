from virtuosoWrapper.annet_o_data_management import annet_o_data_management as dataMgnt
from NetworkEvaluation.evaluation_result import evaluation_result
from nodes import handler_functions
from nodes.nodeEntity import nodeEntity as nodeClass
from Network.network import network
import copy

class handle_entities:
    def __init__(self):
        self.data=dataMgnt()
        self.node_map={}
        self.current_network=""
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
        self.discovered_loss = []
        self.discovered_optimizers = []
        self.input_layers = []
        self.names_into_separator=[]
        self.output_layer=[]
        self.input_layer=[]

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

    def one_network_training(self,session,part_name,opl):
        #There is a case where a second optimizer is used as an initialization optimizer for some variables of the network.
        #This function is used for the case of only one neural network.
        #Thus after the call to find which optimizer belong to the networ,we break the iteration.
        for obj in self.data.annConfiguration.networks[self.current_network].objective.keys():
            #opl=self.find_optimizer()
            (n_name,_,_,_)=\
                self.find_network(self.data.annConfiguration.networks[self.current_network].objective[obj],self.current_network,0,opl)
            break
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


        epoch=0
        batch=0
        optKey=""
        for key in optimizer.keys():
            print("OUTER KEY OPTIMIZER=",key)
            for sess in session.keys():
                print("1992INNER KEY OPTIMIZER=", sess)
                print("session=",sess)
                trSession=session[sess]
                for step in trSession.steps:
                    for opt in step.optimizer:
                        print("Optimizer node name =", optimizer[key].name, " file ", opt)
                        if optimizer[key].name==opt:
                            print("Epoch is ",step.epoch)
                            epoch=step.epoch
                            for ind,input in enumerate(step.inputs):

                                for inp_ in self.data.annConfiguration.networks[n_name].input_layer:
                                    print("1INPUT LAYERS=", " ----- ",
                                          input)
                                    print("2INPUT LAYERS=", " ----- ",
                                          inp_.name)
                                    if input==inp_.name:
                                        print("Batch is ",step.batches[ind])
                                        batch=step.batches[ind]
                                        optKey=key
        IOPipe = handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[n_name], "train")
        #print("LOCO=",optimizer[key].name)
        primary_tr_step = handler_functions.handle_training_single(part_name+"_training_step",n_name,IOPipe,[optimizer[optKey].name],epoch,batch,"")
        tr_session=handler_functions.handle_training_session(part_name+"_training_session",[],primary_tr_step)
        tr_strategy=handler_functions.handle_training_strategy(part_name+"_training_strategy",[tr_session],tr_model)
        self.data.evaluationResult.train_strategy=tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name]=tr_strategy
        self.insert_to_evaluation_pipe(n_name)
        del self.data.annConfiguration.networks[self.current_network]

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
        return 1

    def handle_possible_loss_functions(self):
        for poss in self.possible_loss_function.keys():
            cnt=0
            print("POSSIBLE LOSS FUNCTION ARE:",self.possible_loss_function[poss])
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


    def find_network_input_layer_and_layers(self,output):
        layers = self.data.annConfiguration.networks[self.current_network].layer
        prev=output.name
        net_layers = {}
        net_layers[prev] = output
        print("Start searching from output ",output.name)
        if prev not in layers.keys():
            prev=self.nodes_to_layers[prev]
            #print("Previous translated to ",prev)
        prev=layers[prev].previous_layer
        ret_layer = []
        if layers[output.name].is_input==True:
            ret_layer.append(layers[output.name].placeholder)
            net_layers[layers[output.name].placeholder] = layers[layers[output.name].placeholder]
            net_layers[output.name] = layers[output.name]
        else:
            while prev!=[]:
                temp=[]
                for prev_layer in prev:
                    if layers[prev_layer].is_input==True:
                        print("LOGGING:Discovered input layer = ",prev_layer)
                        ret_layer.append(layers[prev_layer].placeholder)
                        net_layers[layers[prev_layer].placeholder]=layers[layers[prev_layer].placeholder]
                        net_layers[prev_layer] = layers[prev_layer]
                        for elem in layers[prev_layer].previous_layer:
                            temp.append(elem)
                    else:
                        print("LOGGING:Discovered intermediate layer = ",prev_layer)
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

    def insert_network(self,layers,name,inputs,outputs,objective_func,datasets):
        self.data.init_new_network(name)
        print("\n---------------START NEW NETWORK-----------------\n")
        t_layers=copy.deepcopy(layers)
        for input in inputs:
            input_layer=handler_functions.handle_in_layer(layers[input])
            input_layer.placeholder=layers[input].placeholder
            print("LOGGING:["+name+"]Input is =", input," Dataset is = ",input_layer.placeholder)
            self.data.annConfiguration.networks[name].input_layer.append(input_layer)
            del t_layers[input]
        for output in outputs:
            print("LOGGING:["+name+"]Output is = ",output.name)
            output_layer = handler_functions.handle_out_layer(output)
            self.data.annConfiguration.networks[name].output_layer.append(output_layer)
        for layer in t_layers.keys():
            print("LOGGING:["+name+"]Layer is ",layer)
            self.data.annConfiguration.networks[name].layer[layer]=copy.deepcopy(layers[layer])
        if objective_func!="":
            print("LOGGING:["+name+"]Objective func is = ",objective_func.name)
            self.data.annConfiguration.networks[name].objective[objective_func.name]=objective_func
        if datasets.keys()!=[]:
            print("LOGGING:["+name+"]Dataset is = ", datasets)
            self.data.annConfiguration.networks[name].datasets=datasets
        print("\n---------------END NEW NETWORK-----------------\n")

    def create_tr_step(self,optimizer,network,step):
        for opt in optimizer:
            optimizer_=self.data.annConfiguration.networks[self.current_network].optimizer[opt]
            self.data.annConfiguration.networks[network].optimizer[opt]=optimizer_
        input_layer=self.data.annConfiguration.networks[network].input_layer
        print("LOGGING:Input_layer ",[x.name for x in input_layer]," Optimizer ",optimizer)
        IOPipe = handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[network],"train")
        #TODO:Maybe needs check with input for batches,for the moment consider it to be the first one
        tr_step = handler_functions.handle_training_single(step.name, network, IOPipe,optimizer, step.epoch,step.batches[0], step.next)
        return tr_step

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
                            #print("FOUND:\nOptimizer=",optimizer,"\nLname=",elem_,"\ngradient=",gradient)
                            print(optimizer_per_layer)
        for key in optimizer_per_layer.keys():
            str=""
            for x in optimizer_per_layer[key]:
                str=str+x+"\n"
        return (optimizer_per_layer)

    def insert_strategy(self,trSessions):
        tr_model = handler_functions.handle_trained_model(self.data.AnnConfig + "_trained_model")
        layers = self.data.annConfiguration.networks[self.current_network].layer
        counter = 0
        for layer in layers.keys():
            if len(layers[layer].previous_layer) != 0:
                for elem_in in layers[layer].previous_layer:
                    counter += 1
                    weight = handler_functions.handle_weights('W' + str(counter), elem_in, layer)
                    tr_model.add_weight(weight)
        tr_strategy = handler_functions.handle_training_strategy(str(self.current_network) + "_training_strategy", trSessions, tr_model)
        IOPipe = handler_functions.handle_dataset_pipe_1(self.data.annConfiguration.networks[self.current_network], "test")
        self.check_metric(IOPipe, self.current_network)
        self.data.evaluationResult.train_strategy = tr_strategy
        self.data.annConfiguration.training_strategy[tr_strategy.name] = tr_strategy
        self.data.evaluationResult.ann_conf = self.data.annConfiguration

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
                if [ layer_names[0] for x in set(loss.inner_nodes) if layer_names[0]==x]!=[]:
                    del self.data.annConfiguration.networks[self.current_network].layer[layer_names[0]]
                layer_names.remove(layer_names[0])
                print(len(layer_names))

    def find_training(self,sessions):
        res = self.check_multiple_networks()
        optimizer_per_layer = self.find_optimizer()
        optimizer_map = {}
        optimizer_per_layer= self.remove_unused_optimizers(sessions, optimizer_per_layer)
        if res == 0:
            print("LOGGING:Only one network presented.")
            self.one_network_training(sessions, self.current_network,optimizer_per_layer)
            return res
        elif res == -1:
            print("ERROR:Program not a network finally")
            return "ERROR:This tensorflow program is not a network.No objective functions identified"
        objectives = []
        for l in self.data.annConfiguration.networks[self.current_network].objective.keys():
            objectives.append(l)
        net_cnt=0
        allSessions=[]
        for obj in sorted(objectives):
            print("LOGGING:Encountered objective ",
                  self.data.annConfiguration.networks[self.current_network].objective[obj].name)
            obj_func = self.data.annConfiguration.networks[self.current_network].objective[obj]
            (n_name, output, net_cnt, optimizer) = self.find_network(obj_func,self.current_network, net_cnt,optimizer_per_layer)
            print("ASSOCIATE ",n_name," optimizer ",optimizer)
            optimizer_map[n_name]=optimizer
        for trSession in sessions.keys():
            print("LOGGING:Begin searching for session",trSession)
            trSessionElem=sessions[trSession]
            isLoop = False
            for step in trSessionElem.steps:
                if "co_train" in step.name:
                    isLoop=True
            if isLoop==True:
                print("LOGGING:Begin handling session ",trSession," that is a loop session.")
                tr_session=self.handle_training_step(trSessionElem,trSession,optimizer_map,True)
            else:
                print("LOGGING:Begin handling session ", trSession, " that is a normal session.")
                tr_session=self.handle_training_step(trSessionElem,trSession,optimizer_map,False)
            if tr_session!=None:
                allSessions.append(tr_session)
            else:
                print("ERROR:",trSession," had no kind of training steps.Thus not training session added.")
        self.insert_strategy(allSessions)
        del self.data.annConfiguration.networks[self.current_network]
        return 1

    def handle_training_step(self,trSession,trSessionName,optimizer_map,isLoop):
        trSteps=[]
        primary_in_loop_tr_step = ""
        looping_steps = []
        stop_cond=-1
        primary_tr_step=""
        for step in trSession.steps:
            if len(step.optimizer)==0:
                print("ERROR:Return,no optimizer handle for ",trSessionName," it smh after,skip it for now")
            elif len(step.optimizer)>1:
                print("LOGGING:Multiple co training of networks")
                for network in self.data.annConfiguration.networks:
                    if network in optimizer_map.keys():
                        found = True
                        for optimizer in optimizer_map[network]:
                            if optimizer not in step.optimizer:
                                found=False
                                break
                        if found==True:
                            print("STAVRIOS PANERAS IS =",network," for ",step.name)
                            trStep=self.create_tr_step(step.optimizer,network,step)
                            if isLoop==True:
                                if "_co_train" in step.name:
                                    primary_in_loop_tr_step=trStep
                                    stop_cond=step.epoch
                                else:
                                    looping_steps.append(trStep)
                            else:
                                trSteps.append(trStep)
                        else:
                            print(network," OPTIMIZER ",optimizer_map[network]," not in ",step.optimizer)
                    else:
                        print("ERROR:Network ",network," with no optimizer.")
            else:
                print("LOGGING:Only one optimizer-network")
                for network in self.data.annConfiguration.networks:
                    if network in optimizer_map.keys():
                        for optimizer in optimizer_map[network]:
                            if optimizer ==step.optimizer[0]:
                                trStep = self.create_tr_step([optimizer], network, step)
                                primary_tr_step=trStep
                    else:
                        print("ERROR:Network ",network," with no optimizer.")
        if trSteps==[] and looping_steps==[] and primary_in_loop_tr_step=="" and primary_tr_step=="":
            print("ERROR:Unable to find any kind of training steps for ",trSessionName)
            return None
        if primary_in_loop_tr_step!="":
            primary_tr_step=handler_functions.handle_loop(trSessionName+"_training_loop",primary_in_loop_tr_step,looping_steps,stop_cond)
        tr_session = handler_functions.handle_training_session(trSessionName + "_training_session", trSteps,primary_tr_step)
        return tr_session

    def remove_unused_optimizers(self,sessions,opl):
        tmp_opl=opl.copy()
        ret_session=sessions.copy()
        for sess in sessions.keys():
            for trStep in sessions[sess].steps:
                for optimizer in trStep.optimizer:
                    if optimizer in tmp_opl.keys():
                        print("DELETING ",optimizer)
                        del tmp_opl[optimizer]
        for key in tmp_opl.keys():
            del opl[key]
            print("ERROR:DELETED OPTIMIZER NOT FOUND IN FILE TRAINING ",key)
        for optimizer in opl.keys():
            self.data.annConfiguration.networks[self.current_network].optimizer[optimizer].insert_in_annetto()
        return opl

    def find_network(self,obj_func,network,counter,opl):
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
            (tinput,tlayers)=self.find_network_input_layer_and_layers(mem)
            for inp in tinput:
                if inp not in inputs:
                    inputs.append(inp)
            for k,v in tlayers.items():
                layers[k]=v
        n_name = network + "_" + str(counter)
        l_output=[]
        for output in outputs:
            l_output.append(self.data.annConfiguration.networks[self.current_network].layer[output.get_name()])
        self.insert_network(layers,n_name, inputs, layer_outputs, obj_func,datasets)
        counter += 1
        st=""
        for x in layers.keys():
            st=st+"\n"+x
        network_optimizers=[]
        for optimizer in opl.keys():
            found_optimizer=True
            for layer in layers.keys():
                print("searching for layer ", layer, " in ", optimizer," with ",opl[optimizer])
                if layer in opl[optimizer]:
                    network_optimizers.append(optimizer)
                    break
        return (n_name,outputs,counter,set(network_optimizers))
