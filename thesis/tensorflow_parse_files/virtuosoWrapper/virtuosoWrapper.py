import nodes.handler
log_msg1="VirtuosoWrapper:Function to insert new "
log_msg2=" with name "
log_file="log.txt"
def log(str):
    with open("log.txt", "a") as myfile:
        myfile.write(str+"\n")

def insert_ann_graph():
    nodes.handler.entitiesHandler.data.annConfiguration.insert_in_annetto()
    nodes.handler.entitiesHandler.data.evaluationResult.insert_in_annetto()


def new_init_data(name):
    log(log_msg1 + "init_data" + log_msg2 + name)
    nodes.handler.entitiesHandler.data.init_data(name)

def new_init_new_network(name):
    log(log_msg1 + "init network" + log_msg2 + name)
    nodes.handler.entitiesHandler.current_network =name
    nodes.handler.entitiesHandler.data.init_new_network(name)

def new_init_new_evaluation(name,network):
    log(log_msg1 + "init evaluation" + log_msg2 + name+":"+network)
    nodes.handler.entitiesHandler.current_evaluation=name+"_val"
    nodes.handler.entitiesHandler.data.init_new_network_evaluation(name,network)

def new_ann_configuration(name):
    log(log_msg1+"ann configuration"+log_msg2+name)
    nodes.handler.entitiesHandler.data.insert_annConfiguration(name)

def new_network(name):
    log(log_msg1+"network"+log_msg2+ name)
    nodes.handler.entitiesHandler.data.insert_network(name + "_net")

def new_training_strategy(name):
    log(log_msg1+"training_strategy"+log_msg2+name )
    nodes.handler.entitiesHandler.data.insert_training_strategy(name+"_training_strategy")

def new_has_network(network):
    log(log_msg1 + "has network" + log_msg2 + network)
    nodes.handler.entitiesHandler.data.insert_has_network(network)

def new_has_tr_strategy(trStrategy):
    log(log_msg1 + "training_strategy" + log_msg2 + trStrategy)
    nodes.handler.entitiesHandler.data.insert_has_training_strategy(trStrategy)

'''
def new_has_layer(layer):
    log(log_msg1 + "has layer" + log_msg2 + layer+"->"+nodes.handler.entitiesHandler.current_network)
    nodes.handler.entitiesHandler.data.insert_hasLayer(layer,nodes.handler.entitiesHandler.current_network)
    # ata.insert_hasLayer(layer,urrent_network)
'''
def new_next_layer(name ,next_layer):
    '''
    if len(next_layer)>1:
        import sys
        print("WRONG NUMBER OF NEXT LAYERS,MORE THAN ONE")
    if len(next_layer)==0:
        print("empty next")
        layer=""
    else:
        layer =next_layer[0]
    '''
    print("name="+name)
    log(log_msg1 + "next layer" + log_msg2 + name+"->"+next_layer)
    nodes.handler.entitiesHandler.data.insert_nextLayer(name,next_layer)

def new_previous_layer(name ,previous_layer):
    '''
    if len(previous_layer)>1:
        print("WRONG NUMBER OF PREVIOUS LAYERS,MORE THAN ONE=",previous_layer)
    if len(previous_layer)==0:
        print("empty prev")
        layer=""
    else:
        layer =previous_layer[0]
    '''
    log(log_msg1 + "previous layer" + log_msg2 + name+"->"+previous_layer)
    nodes.handler.entitiesHandler.data.insert_prevLayer(name, previous_layer)

def new_named_individual(name):
    log(log_msg1 + "named individual" + log_msg2 + name)
    nodes.handler.entitiesHandler.data.insert_named_indiv(name)

def new_has_activation(name ,activation):
    print(name," ",activation)
    log(log_msg1 + "has activation" + log_msg2 + name+"->"+activation)
    nodes.handler.entitiesHandler.data.insert_hasActivation(name,activation)

def new_type(name ,type):
    print(name)
    log(log_msg1 + "type" + log_msg2 + name+"->"+type)
    nodes.handler.entitiesHandler.data.insert_type(name,type)

def new_has_bias(layer,value):
   log(log_msg1 + "has bias" + log_msg2 + layer + "->" + str(value))
   nodes.handler.entitiesHandler.data.insert_has_bias(layer,value)

def layer_num_units(layer,num):
    log(log_msg1 + "layer num" + log_msg2 + layer + "->" + str(num))
    nodes.handler.entitiesHandler.data.insert_num_of_units(layer, num)

def new_network_has_layer(network,layer):
    log(log_msg1 + "has layer" + log_msg2 + network + "->" + layer)
    nodes.handler.entitiesHandler.data.insert_has_layer(network, layer)

def new_has_loss(cost,loss):
    log(log_msg1 + "has loss" + log_msg2 + cost + "->" + loss)
    nodes.handler.entitiesHandler.data.insert_has_loss(cost, loss)

def new_has_cost(objective,cost):
    log(log_msg1 + "has cost" + log_msg2 + objective + "->" + cost)
    nodes.handler.entitiesHandler.data.insert_has_cost(objective, cost)

def new_network_has_objective(network,objective):
    log(log_msg1 + "has objective" + log_msg2 + network + "->" + objective)
    nodes.handler.entitiesHandler.data.insert_has_objective(network, objective)

def new_trains_network(training,network):
    log(log_msg1 + "trains network" + log_msg2 + training + "->" + network)
    nodes.handler.entitiesHandler.data.insert_trains_network(training, network)


def new_io_pipe(training,IOPipe):
    log(log_msg1 + "hasIOPipe" + log_msg2 + training + "->" + IOPipe)
    nodes.handler.entitiesHandler.data.insert_io_pipe(training, IOPipe)

def new_has_optimizer(trStep,optimizer):
    log(log_msg1 + " has optimizer " + log_msg2 + trStep + "->" + optimizer)
    nodes.handler.entitiesHandler.data.insert_optimizer(trStep, optimizer)

def new_joins_layer(pipe,layer):
    log(log_msg1 + "joins layer" + log_msg2 + pipe + "->" + layer)
    nodes.handler.entitiesHandler.data.insert_joins_layer(pipe, layer)

def new_joins_dataset(pipe,dataset):
    log(log_msg1 + "joins dataset" + log_msg2 + pipe + "->" + dataset)
    nodes.handler.entitiesHandler.data.insert_joins_dataset(pipe, dataset)

def new_learning_rate(optimizer,learning_rate):
    log(log_msg1 + "new_learning_rate" + log_msg2 + optimizer + "->" + str(learning_rate))
    nodes.handler.entitiesHandler.data.insert_learning_rate(optimizer, learning_rate)

def new_trained_in_layer(weight,layer):
    log(log_msg1 + "trained in" + log_msg2 + weight + "->" + str(layer))
    nodes.handler.entitiesHandler.data.insert_trained_in(weight, layer)

def new_trained_out_layer(weight,layer):
    log(log_msg1 + "trained out" + log_msg2 + weight + "->" + str(layer))
    nodes.handler.entitiesHandler.data.insert_trained_out(weight, layer)

def new_has_weights(model,weight):
    log(log_msg1 + "has weights" + log_msg2 + model + "->" + str(weight))
    nodes.handler.entitiesHandler.data.insert_has_weight(model, weight)

def new_evaluates_ann_conf(evRes,ann_conf):
    log(log_msg1 + "evaluates ann conf" + log_msg2 + evRes + "->" + str(ann_conf))
    nodes.handler.entitiesHandler.data.insert_eval_ann_conf(evRes, ann_conf)

def new_evaluates_network(evRes,network):
    log(log_msg1 + "evaluates network" + log_msg2 + evRes + "->" + str(network))
    nodes.handler.entitiesHandler.data.insert_eval_network(evRes, network)


def new_evaluates_using_io(evRes,pipe):
    log(log_msg1 + "evaluates using io" + log_msg2 + evRes + "->" + str(pipe))
    nodes.handler.entitiesHandler.data.insert_eval_using_io(evRes, pipe)


def new_has_metric(evRes,metric):
    log(log_msg1 + "has metric" + log_msg2 + evRes + "->" + str(metric))
    nodes.handler.entitiesHandler.data.insert_has_metric(evRes, metric)


def new_with_tr_strategy(evRes,trStrat):
    log(log_msg1 + "has training strategy" + log_msg2 + evRes + "->" + str(trStrat))
    nodes.handler.entitiesHandler.data.insert_with_tr_str(evRes, trStrat)

def new_has_prim_tr_session(trStr,sess):
    log(log_msg1 + "has primary training session" + log_msg2 + trStr + "->" + str(sess))
    nodes.handler.entitiesHandler.data.insert_has_prim_tr_session(trStr, sess)

def new_trained_model(trStr, model):
    log(log_msg1 + "has trained model" + log_msg2 + trStr + "->" + str(model))
    nodes.handler.entitiesHandler.data.insert_has_trained_model(trStr, model)


def new_batch_size(annConf,batch):
    log(log_msg1 + "batch size" + log_msg2 + annConf + "->" + str(batch))
    nodes.handler.entitiesHandler.data.insert_batch(annConf, batch)

def new_epoch_num(annConf,epoch):
    log(log_msg1 + "epoch num" + log_msg2 + annConf + "->" + str(epoch))
    nodes.handler.entitiesHandler.data.insert_epoch(annConf, epoch)


def new_input_layer(network,input):
    log(log_msg1 + "input layer" + log_msg2 + network + "->" + str(input))
    nodes.handler.entitiesHandler.data.insert_input_layer(network, input)

def new_output_layer(network, output):
    log(log_msg1 + "output layer" + log_msg2 + network + "->" + str(output))
    nodes.handler.entitiesHandler.data.insert_output_layer(network, output)


def new_has_training_step(session,step):
    log(log_msg1 + "training step" + log_msg2 + session + "->" + str(step))
    nodes.handler.entitiesHandler.data.insert_step(session, step)

def new_has_primary_training_step(session,prStep):
    log(log_msg1 + "primary training step" + log_msg2 + session + "->" + str(prStep))
    nodes.handler.entitiesHandler.data.insert_primary_step(session, prStep)
