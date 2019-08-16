import nodes.handler
log_msg1="VirtuosoWrapper:Function to insert new "
log_msg2=" with name "
log_file=""
map_to_entry_name={}
not_added_yet=[]
same_layer=[]
counter=0
file_counter=0
def reset_vars():
    global counter
    global map_to_entry_name
    global not_added_yet
    map_to_entry_name = {}
    not_added_yet = []

    counter = 0

def log(str):
    global counter
    if counter==0:
        with open(log_file, "w") as myfile:
            myfile.write(str+"\n")
        counter+=1
    else:
        with open(log_file, "a") as myfile:
            myfile.write(str+"\n")

def insert_ann_graph():
    nodes.handler.entitiesHandler.data.annConfiguration.insert_in_annetto()
    nodes.handler.entitiesHandler.data.evaluationResult.insert_in_annetto()


def new_init_data(name):
    #log(log_msg1 + "init_data" + log_msg2 + name)
    global file_counter
    if name not in map_to_entry_name.keys():
        map_to_entry_name[name]=name+str(file_counter)
        file_counter+=1
    nodes.handler.entitiesHandler.data.init_data(map_to_entry_name[name])

def new_init_new_network(name):
    #log(log_msg1 + "init network" + log_msg2 + name)
    nodes.handler.entitiesHandler.current_network =name
    nodes.handler.entitiesHandler.data.init_new_network(name)

def new_init_new_evaluation(name,network):
    #log(log_msg1 + "init evaluation" + log_msg2 + name+":"+network)
    nodes.handler.entitiesHandler.current_evaluation=name+"_val"
    nodes.handler.entitiesHandler.data.init_new_network_evaluation(name,network)

def new_ann_configuration(name):
    #log(log_msg1+"ann configuration"+log_msg2+name)
    nodes.handler.entitiesHandler.data.insert_annConfiguration(name)

def new_network(name):
    #log(log_msg1+"network"+log_msg2+ name)
    nodes.handler.entitiesHandler.data.insert_network(map_to_entry_name[name])

def new_training_strategy(name):
    #log(log_msg1+"training_strategy"+log_msg2+name )
    nodes.handler.entitiesHandler.data.insert_training_strategy(map_to_entry_name[name])

def new_has_network(network):
    #log(log_msg1 + "has network" + log_msg2 + network)
    nodes.handler.entitiesHandler.data.insert_has_network(map_to_entry_name[network])

def new_has_tr_strategy(trStrategy):
    #log(log_msg1 + "training_strategy" + log_msg2 + trStrategy)
    nodes.handler.entitiesHandler.data.insert_has_training_strategy(map_to_entry_name[trStrategy])

def new_next_layer(name ,next_layer):
    global file_counter
    if next_layer not in map_to_entry_name.keys():
        map_to_entry_name[next_layer]=next_layer+str(file_counter)
        not_added_yet.append(next_layer)
        file_counter+=1
    #log(log_msg1 + "next layer" + log_msg2 + name+"->"+next_layer)
    nodes.handler.entitiesHandler.data.insert_nextLayer(map_to_entry_name[name],map_to_entry_name[next_layer])

def new_previous_layer(name ,previous_layer):
    global file_counter
    #log(log_msg1 + "previous layer" + log_msg2 + name+"->"+previous_layer)
    if previous_layer not in map_to_entry_name.keys():
        not_added_yet.append(previous_layer)
        map_to_entry_name[previous_layer]=previous_layer+str(file_counter)
        file_counter+=1
    nodes.handler.entitiesHandler.data.insert_prevLayer(map_to_entry_name[name], map_to_entry_name[previous_layer])

def new_named_individual(name):
    global file_counter
    #log(log_msg1 + "named individual" + log_msg2 + name)
    isSameLayer=""
    if name in not_added_yet:
        nodes.handler.entitiesHandler.data.insert_named_indiv(map_to_entry_name[name])
    elif name not in map_to_entry_name.keys():
        map_to_entry_name[name] = name + str(file_counter)
        file_counter += 1
        nodes.handler.entitiesHandler.data.insert_named_indiv(map_to_entry_name[name])
        return 0
    elif name in map_to_entry_name.keys():
        print("LOGGING:Individual ",name," is re-used")
        return 1




def new_has_activation(name ,activation):
    #log(log_msg1 + "has activation" + log_msg2 + name+"->"+activation)
    nodes.handler.entitiesHandler.data.insert_hasActivation(map_to_entry_name[name],map_to_entry_name[activation])

def new_type(name ,type):
    #log(log_msg1 + "type" + log_msg2 + name+"->"+type)
    nodes.handler.entitiesHandler.data.insert_type(map_to_entry_name[name],type)

def new_has_bias(layer,value):
   #log(log_msg1 + "has bias" + log_msg2 + layer + "->" + str(value))
   nodes.handler.entitiesHandler.data.insert_has_bias(map_to_entry_name[layer],value)

def layer_num_units(layer,num):
    #log(log_msg1 + "layer num" + log_msg2 + layer + "->" + str(num))
    nodes.handler.entitiesHandler.data.insert_num_of_units(map_to_entry_name[layer],num)

def new_network_has_layer(network,layer):
    #log(log_msg1 + "has layer" + log_msg2 + network + "->" + layer)
    global file_counter
    # log(log_msg1 + "named individual" + log_msg2 + name)
    if network not in map_to_entry_name.keys():
        map_to_entry_name[network] = network + str(file_counter)
        file_counter += 1
    if layer not in map_to_entry_name.keys():
        map_to_entry_name[layer] = layer + str(file_counter)
        file_counter += 1
    nodes.handler.entitiesHandler.data.insert_has_layer(map_to_entry_name[network], map_to_entry_name[layer])

def new_has_loss(cost,loss):
    #log(log_msg1 + "has loss" + log_msg2 + cost + "->" + loss)
    nodes.handler.entitiesHandler.data.insert_has_loss(map_to_entry_name[cost], map_to_entry_name[loss])

def new_has_cost(objective,cost):
    #log(log_msg1 + "has cost" + log_msg2 + objective + "->" + cost)
    nodes.handler.entitiesHandler.data.insert_has_cost(map_to_entry_name[objective], map_to_entry_name[cost])

def new_network_has_objective(network,objective):
    #log(log_msg1 + "has objective" + log_msg2 + network + "->" + objective)
    nodes.handler.entitiesHandler.data.insert_has_objective(map_to_entry_name[network], map_to_entry_name[objective])

def new_trains_network(training,network):
    #log(log_msg1 + "trains network" + log_msg2 + training + "->" + network)
    '''
    global file_counter
    # log(log_msg1 + "named individual" + log_msg2 + name)
    if training not in map_to_entry_name.keys():
        map_to_entry_name[training] = training + str(file_counter)
        file_counter += 1
    if network not in map_to_entry_name.keys():
        map_to_entry_name[network] = network + str(file_counter)
        file_counter += 1
    '''
    nodes.handler.entitiesHandler.data.insert_trains_network(map_to_entry_name[training], map_to_entry_name[network])


def new_io_pipe(training,IOPipe):
    #log(log_msg1 + "hasIOPipe" + log_msg2 + training + "->" + IOPipe)
    nodes.handler.entitiesHandler.data.insert_io_pipe(map_to_entry_name[training], map_to_entry_name[IOPipe])

def new_has_optimizer(trStep,optimizer):
    #log(log_msg1 + " has optimizer " + log_msg2 + trStep + "->" + optimizer)
    nodes.handler.entitiesHandler.data.insert_optimizer(map_to_entry_name[trStep], map_to_entry_name[optimizer])

def new_joins_layer(pipe,layer):
    #log(log_msg1 + "joins layer" + log_msg2 + pipe + "->" + layer)
    nodes.handler.entitiesHandler.data.insert_joins_layer(map_to_entry_name[pipe], map_to_entry_name[layer])

def new_joins_dataset(pipe,dataset):
    #log(log_msg1 + "joins dataset" + log_msg2 + pipe + "->" + dataset)
    nodes.handler.entitiesHandler.data.insert_joins_dataset(map_to_entry_name[pipe], map_to_entry_name[dataset])

def new_learning_rate(optimizer,learning_rate):
    #log(log_msg1 + "new_learning_rate" + log_msg2 + optimizer + "->" + str(learning_rate))
    nodes.handler.entitiesHandler.data.insert_learning_rate(map_to_entry_name[optimizer],learning_rate)

def new_trained_in_layer(weight,layer):
    #log(log_msg1 + "trained in" + log_msg2 + weight + "->" + str(layer))
    nodes.handler.entitiesHandler.data.insert_trained_in(map_to_entry_name[weight], map_to_entry_name[layer])

def new_trained_out_layer(weight,layer):
    #log(log_msg1 + "trained out" + log_msg2 + weight + "->" + str(layer))
    nodes.handler.entitiesHandler.data.insert_trained_out(map_to_entry_name[weight], map_to_entry_name[layer])

def new_has_weights(model,weight):
    #log(log_msg1 + "has weights" + log_msg2 + model + "->" + str(weight))
    nodes.handler.entitiesHandler.data.insert_has_weight(map_to_entry_name[model], map_to_entry_name[weight])

def new_evaluates_ann_conf(evRes,ann_conf):
    #log(log_msg1 + "evaluates ann conf" + log_msg2 + evRes + "->" + str(ann_conf))
    nodes.handler.entitiesHandler.data.insert_eval_ann_conf(map_to_entry_name[evRes], map_to_entry_name[ann_conf])

def new_evaluates_network(evRes,network):
    #log(log_msg1 + "evaluates network" + log_msg2 + evRes + "->" + str(network))
    nodes.handler.entitiesHandler.data.insert_eval_network(map_to_entry_name[evRes], map_to_entry_name[network])


def new_evaluates_using_io(evRes,pipe):
    #log(log_msg1 + "evaluates using io" + log_msg2 + evRes + "->" + str(pipe))
    nodes.handler.entitiesHandler.data.insert_eval_using_io(map_to_entry_name[evRes], map_to_entry_name[pipe])


def new_has_metric(evRes,metric):
    #log(log_msg1 + "has metric" + log_msg2 + evRes + "->" + str(metric))
    nodes.handler.entitiesHandler.data.insert_has_metric(evRes, metric)


def new_with_tr_strategy(evRes,trStrat):
    #log(log_msg1 + "has training strategy" + log_msg2 + evRes + "->" + str(trStrat))
    nodes.handler.entitiesHandler.data.insert_with_tr_str(map_to_entry_name[evRes], map_to_entry_name[trStrat])

def new_has_prim_tr_session(trStr,sess):
    #log(log_msg1 + "has primary training session" + log_msg2 + trStr + "->" + str(sess))
    nodes.handler.entitiesHandler.data.insert_has_prim_tr_session(map_to_entry_name[trStr], map_to_entry_name[sess])

def new_trained_model(trStr, model):
    #log(log_msg1 + "has trained model" + log_msg2 + trStr + "->" + str(model))
    nodes.handler.entitiesHandler.data.insert_has_trained_model(map_to_entry_name[trStr], map_to_entry_name[model])


def new_batch_size(annConf,batch):
    #log(log_msg1 + "batch size" + log_msg2 + annConf + "->" + str(batch))
    nodes.handler.entitiesHandler.data.insert_batch(map_to_entry_name[annConf], batch)

def new_epoch_num(annConf,epoch):
    #log(log_msg1 + "epoch num" + log_msg2 + annConf + "->" + str(epoch))
    nodes.handler.entitiesHandler.data.insert_epoch(map_to_entry_name[annConf],epoch)


def new_input_layer(network,input):
    #log(log_msg1 + "input layer" + log_msg2 + network + "->" + str(input))
    nodes.handler.entitiesHandler.data.insert_input_layer(map_to_entry_name[network], map_to_entry_name[input])

def new_output_layer(network, output):
    #log(log_msg1 + "output layer" + log_msg2 + network + "->" + str(output))
    nodes.handler.entitiesHandler.data.insert_output_layer(map_to_entry_name[network], map_to_entry_name[output])


def new_has_training_step(session,step):
    #log(log_msg1 + "training step" + log_msg2 + session + "->" + str(step))
    nodes.handler.entitiesHandler.data.insert_step(map_to_entry_name[session], map_to_entry_name[step])

def new_has_primary_training_step(session,prStep):
    #log(log_msg1 + "primary training step" + log_msg2 + session + "->" + str(prStep))
    nodes.handler.entitiesHandler.data.insert_primary_step(map_to_entry_name[session], map_to_entry_name[prStep])


def new_has_regularizer(cf,regularizer):
    nodes.handler.entitiesHandler.data.insert_has_regularizer(map_to_entry_name[cf], map_to_entry_name[regularizer])

def new_same_layer(layer,same_layer):
    nodes.handler.entitiesHandler.data.insert_same_layer(map_to_entry_name[layer],map_to_entry_name[same_layer])

def new_has_stop_cond(loop,cond):
    nodes.handler.entitiesHandler.data.insert_has_stop_cond(map_to_entry_name[loop], map_to_entry_name[cond])

def new_has_primary_loop(loop,primaryLoop):
    nodes.handler.entitiesHandler.data.insert_primary_loop(map_to_entry_name[loop], map_to_entry_name[primaryLoop])

def new_has_looping_step(loop,loopingStep):
    nodes.handler.entitiesHandler.data.insert_looping_step(map_to_entry_name[loop], map_to_entry_name[loopingStep])

def new_has_primary_loop(loop,loopingStep):
    nodes.handler.entitiesHandler.data.insert_primary_looping_step(map_to_entry_name[loop], map_to_entry_name[loopingStep])

def new_num_of_iterations(name,iterations):
    nodes.handler.entitiesHandler.data.insert_stop_condition_number(map_to_entry_name[name],iterations)