from functions.loss.binary_cross_entropy import binary_cross_entropy
from functions.loss.categorical_cross_entropy import categorical_cross_entropy
from layers.hidden.activation.simple_layer import simple_layer
from layers.hidden.activation.conv2d_layer import conv2d_layer
from layers.hidden.activation.deconv2d_layer import deconv2d_layer
from layers.hidden.aggregation.maxpool_layer import maxpool_layer
from layers.hidden.aggregation.concat_layer import concat_layer
from layers.hidden.modification.dropout_layer import dropout_layer
from layers.hidden.modification.batch_norm_layer import batch_norm_layer
from layers.hidden.modification.flatten_layer import flatten_layer
from functions.activation.non_diff.relu import relu
from functions.activation.regularization.dropout import dropout
from functions.metric.accuracy import accuracy
from functions.metric.mean_squared_error import mean_squared_error
from functions.activation.smooth.softmax import softmax
from  layers.hidden.activation.rnn_layer.lstm_layer import lstm_layer
from  layers.hidden.activation.rnn_layer.gru_layer import gru_layer
from functions.metric.accuracy import accuracy
from NetworkEvaluation.network_evaluation import network_evaluation
from DatasetPipe.dataset_pipe import dataset_pipe
from Algorithm.TrainingOptimizer.gradient_descent import gradient_descent
from Algorithm.TrainingOptimizer.rms_prop import rms_prop
from Algorithm.TrainingOptimizer.adam import adam
from Algorithm.WeightInitialization.random_initialization import random_initialization
from TrainedModel.trained_model import trained_model
from TrainedWeights.trained_weights import trained_weights
from TrainingStep.NetworkSpecific.training_single import training_single
from TrainingStep.TrainingLoop.training_loop import training_loop
from TrainingSession.training_session import training_session
from TrainingStrategy.training_strategy import training_strategy
import nodes.handler
from functions.cost.cost_function import cost_function
from functions.loss.mse import mse
from functions.objective.MinimizeObjective.minimize_objective import minimize_objective
from layers.InOutLayer.InputLayer.input_layer import input_layer
from layers.InOutLayer.OutputLayer.output_layer import output_layer
from layers.InOutLayer.in_out_layer import in_out_layer
from Dataset.label_set import label_set

def handle_lstm( node,name):
    return lstm_layer(node,name)

def handle_gru( node,name):
    return gru_layer(node,name)

def handle_rms_prop(node,name):
    return rms_prop(node,name)

def handle_gradient_descent( node,name):
    return gradient_descent(node,name)

def check_if_acc(node):
    children=[x for x in node.get_inputs()]
    for x in children:
        if x.get_op()=="Placeholder" \
                or x.get_op()=="ArgMax"\
                or x.get_op()=="Softmax":
            return True
        elif x.get_op() in nodes.handler.entitiesHandler.intermediate_operations:
            return check_if_acc(x)
    return False

def handle_mse_metric(node):
    return mean_squared_error(node)

def handle_accuracy(node):
    res=check_if_acc(node)
    if res==True:
        return accuracy(node)
    else:
        return None

def handle_dataset_pipe_1(network,type):
    dp_list=[]
    if type=="test":
        for ind,elem in enumerate(network.output_layer):
            labelSet=""
            node=""
            dataset_name=""
            if elem.name in network.datasets.keys():
                dataset_name=network.datasets[elem.name].get_name()
                node=network.datasets[elem.name]
            print("LOGGING:Found test dataset name ",dataset_name)
            labelSet=label_set(node)
            dp_list.append(dataset_pipe(elem.node, elem, type, str(ind),labelSet))
    if type=="train":
        for ind,elem in enumerate(network.input_layer):
            dataset_name=elem.name
            print("LOGGING:Found train dataset name ",dataset_name)
            labelSet=label_set(elem.node)
            dp_list.append(dataset_pipe(elem.node, elem, type, str(ind),labelSet))
    return dp_list

def handle_sparse_cross_entropy(node,c_name):
    for dim in node.get_output():
        for i,elem in enumerate(dim.dim):
            if elem.size != -1:
                if elem.size > 2:
                    return cost_function(c_name, categorical_cross_entropy(node.get_name(), node))
                else:
                    return cost_function(c_name, binary_cross_entropy(node.get_name(), node))

def handle_random_normal(node):
    return

def handle_softmax( node):
    return softmax(node.get_name(), node.node_obj)

def handle_adam(node,name):
    return adam(node,name)

def handle_sigmoid_entropy(node,c_name,name):
    #TODO SIGMOID
    return cost_function(c_name, categorical_cross_entropy(name, node,True))

def handle_cross_entropy(node,name,c_name):
    new_tmp=[]
    nm = nodes.handler.entitiesHandler.node_map
    outer_inputs=[]
    for node_name in nodes.handler.entitiesHandler.node_map.keys():
        if (name in node_name.split("/") or all(x in node_name.split("/")) for x in name.split("/")) and 'gradients' not in node_name:
            for inp in nm[node_name].get_inputs():
                if name not in inp.get_name():
                    outer_inputs.append(inp)
    while outer_inputs!=[]:
        for inp in outer_inputs:
            #print("Searching for ",inp.get_name()," operation ",inp.get_op())
            if inp.get_op()== "Placeholder":
                for dim in inp.get_attr()["shape"].shape.dim:
                    if dim.size != -1:
                        if dim.size > 2:
                            return cost_function(c_name, categorical_cross_entropy(name, node))
                        else:
                            return cost_function(c_name, binary_cross_entropy(name, node))
                print("ERROR:Found placeholder,but the sizes are all negatives.")
                return cost_function(c_name, categorical_cross_entropy(name, node))
            elif inp.get_op() in nodes.handler.entitiesHandler.intermediate_operations:
                for inp_ in inp.get_inputs():
                    new_tmp.append(inp_)
        outer_inputs=new_tmp
        new_tmp=[]
    return ""

def handle_flatten(node,name):
    return flatten_layer(node,name)

def handle_pow(node,network):
    children=node.get_inputs()
    node_obj=None
    sub_elem=None
    sub=False
    power_of_two=False
    possible=[]
    while children!=[]:
        tmp=[]
        for elem in children:
            if elem.get_op()=="Sub":
                node_obj=elem
                sub_elem=elem
                sub=True
            elif elem.get_op()=="Const":
                num=elem.node_obj.attr["value"].tensor.float_val
                if int(num[0])==2:
                    power_of_two=True
            elif elem.get_op() in nodes.handler.entitiesHandler.intermediate_operations:
                for el in elem.get_inputs():
                    tmp.append(el)
        if sub==True and power_of_two==True:
            return handle_mean_square_error(node_obj,network,node.get_name())
        children=tmp
    else:
        if sub_elem!=None:
            children=sub_elem.get_inputs()
            possible=[]
            while children != []:
                tmp=[]
                for elem in children:
                    if elem.get_op()=="Placeholder":
                        return handle_mean_square_error(sub_elem,network,node.get_name())
                    elif elem.get_op() in nodes.handler.entitiesHandler.intermediate_operations:
                        for el in elem.get_inputs():
                            tmp.append(el)
                    else:
                        possible.append(elem.get_name())
                children = tmp
        #Check if mse is applied to the output of two layers.
        if len(possible)==2:
            nodes.handler.entitiesHandler.possible_loss_function[node.get_name()]=[]
            for child in possible:
                nodes.handler.entitiesHandler.possible_loss_function[node.get_name()].append(child)
        return ""

def handle_neg_for_log(node):
    inputs=node.get_inputs()
    while inputs!=[]:
        tmp=[]
        for input in inputs:
            if input.get_op()=="Log":
                return handle_log(node)
            else:
                inner_inputs=input.get_inputs()
                res=0
                for inner_input in inner_inputs:
                    if inner_input.get_op()=="Log":
                        res+=1
                if res==2:
                    return handle_log(node)
            if input.get_op() in nodes.handler.entitiesHandler.intermediate_operations or input.get_op()=="Add":
                for elem in input.get_inputs():
                    tmp.append(elem)
        inputs=tmp

    return ""

def handle_log(node):
    return cost_function("_cost",categorical_cross_entropy(node.get_name(),node))

def handle_mean_square_error(node,network,name):
    #TODO:FIX FOR SIMPLE/SQUARE MEAN
    mean=mse(node,name)
    return cost_function(network+"_cost",mean)

def check_for_bias(name):
    nm = nodes.handler.entitiesHandler.node_map
    for node_name in nodes.handler.entitiesHandler.node_map.keys():
        #out_name = nm[node_name].get_name()
        if nm[node_name].search_inputs(name) == True and "gradients" not in node_name:
            if nm[node_name].get_op() == "BiasAdd" or nm[node_name].get_op()=="Add":
                return node_name
            else:
                print("ERROR:Conv2d node with name ", name, "should have biasAdd following,this does not.",node_name)
                return -1

def handle_batch_norm(node):
    return batch_norm_layer(node,node.get_name())

def handle_conv2d(node):
    nodeReturn = check_for_bias(node.get_name())
    if nodeReturn!=-1:
        if node.get_inputs()[0].get_op() in nodes.handler.entitiesHandler.variable_operations:
            nodeReturn = conv2d_layer(node.get_inputs()[0], node.get_inputs()[1], node,nodeReturn)
        else:
            nodeReturn = conv2d_layer(node.get_inputs()[1], node.get_inputs()[0], node,nodeReturn)
        return nodeReturn
    return None

def handle_deconv2d(node):
    nodeReturn = check_for_bias(node.get_name())
    if nodeReturn!=-1:
        if node.get_inputs()[0].get_op() in nodes.handler.entitiesHandler.variable_operations:
            nodeReturn = deconv2d_layer(node.get_inputs()[0], node.get_inputs()[1], node,nodeReturn)
        else:
            nodeReturn = deconv2d_layer(node.get_inputs()[1], node.get_inputs()[0], node,nodeReturn)
        return nodeReturn
    return None

def handle_trained_model(name):
    return trained_model(name)

def handle_training_single(name,network,IOPipe,optimizer,epochs,batch,nextTrStep):
    return training_single(name,network,IOPipe,optimizer,epochs,batch,nextTrStep)

def handle_training_strategy(name,session,model):
    return training_strategy(name,session,model)

def find_input_or_output_placeholder(node):
    nm = nodes.handler.entitiesHandler.node_map
    for name in nm.keys():
        if nm[name].search_inputs(node.get_name())==True:
            if nm[name].get_op() not in nodes.handler.entitiesHandler.intermediate_operations:
                for lname in nodes.handler.entitiesHandler.loss_functions:
                    if lname in nm[name].get_name():
                        return output_layer(node)
            else:
                return find_input_or_output_placeholder(nm[name])
    return input_layer(node)



def handle_in_out_layer(node):
    return in_out_layer(node)

def handle_out_layer(layer):
    return output_layer(layer)

def handle_in_layer(layer):
    return input_layer(layer)

def handle_training_session(name,trStep,primaryTrainingStep):
    return training_session(name,trStep,primaryTrainingStep)

def handle_loop(name, trSteps,looping_steps,cond):
    return training_loop(name, trSteps,looping_steps,cond)

def handle_mul_as_cross_entropy(node,c_name):
    children = node.get_inputs()
    log = False
    placeholder = False
    type = None
    for elem in children:
        if elem.get_op() == "Log":
            log = True
        if elem.get_op() == "Placeholder":
            placeholder=True
            for dim in elem.get_attr()["shape"].shape.dim:
                if dim.size != -1:
                    if dim.size > 2:
                        type="c"
                    else:
                        type="b"
    if log == True and placeholder == True:
        if type=="b":
            return cost_function(c_name, binary_cross_entropy(node.get_name(), node))
        else:
            return cost_function(c_name, categorical_cross_entropy(node.get_name(), node))
    else:
        return ""

def handle_objective(name,cost):
    return minimize_objective(name,cost)

def handle_weights(name,in_,out_):
    return trained_weights(name,in_,out_)

def handle_relu( node):
    return relu(node.get_name(), node, node.get_op())

def handle_maxpool( node):
    return maxpool_layer(node)

def handle_concat(node):
    return concat_layer(node)

def handle_dropout( node,name):
    return dropout_layer(node,name)

def check_simple_layer( node,name):
    if "random_" in name or "normal" in name:
        return (False, [])

    #TODO:What happens if no bias in the node???
    if len(node.inputs) == 2:
        if node.inputs[0].get_op() == "MatMul" or node.inputs[0].get_op() == "Identity" or node.inputs[
            0].get_op() == "Placeholder" or node.inputs[0].get_op() == "Mul":
            if node.inputs[1].get_op() == "MatMul" or node.inputs[1].get_op() == "Identity" or node.inputs[
                1].get_op() == "Placeholder" or node.inputs[0].get_op() == "Mul":
                if node.inputs[0].get_op() == "MatMul" or node.inputs[0].get_op() == "Mul":
                    matMul = node.inputs[0]
                    w = ""
                    x = ""
                    if len(matMul.get_inputs())>=2:
                        if matMul.get_inputs()[0].get_op() == "Identity" or matMul.get_inputs()[
                            0].get_op() == "Placeholder":
                            x = matMul.get_inputs()[0]
                            w = matMul.get_inputs()[1]
                        else:
                            w = matMul.get_inputs()[0]
                            x = matMul.get_inputs()[1]
                        bias = node.inputs[1]
                        node = simple_layer(w, x, matMul, bias, node,name)
                    else:
                        print("LOGGING:multiplication with name ",matMul.get_name()," has less than 2 inpus.")
                        return (False,[])
                else:
                    matMul = node.inputs[1]
                    if len(matMul.get_inputs())>=2:
                        if matMul.get_inputs()[0].get_op() == "Identity" or matMul.get_inputs()[
                            0].get_op() == "Placeholder":
                            x = matMul.get_inputs()[0]
                            w = matMul.get_inputs()[1]
                        else:
                            w = matMul.get_inputs()[0]
                            x = matMul.get_inputs()[1]
                        bias = node.inputs[0]
                        node = simple_layer(w, x, matMul, bias, node,name)
                    else:
                        print("ERROR:matMul with name ",matMul.get_name()," has less than 2 inpus.")
                        return (False,[])
                return (True, node)
    else:
        print("ERROR:node ",node.get_name()," has less than 2 inpus.")
    return (False, [])