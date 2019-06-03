from functions.loss.binary_cross_entropy import binary_cross_entropy
from functions.loss.categorical_cross_entropy import categorical_cross_entropy
from loss_function import loss_function
from layers.hidden.activation.simple_layer import simple_layer
from layers.hidden.activation.conv2d_layer import conv2d_layer
from layers.hidden.activation.deconv2d_layer import deconv2d_layer
from layers.hidden.aggregation.maxpool_layer import maxpool_layer
from layers.hidden.aggregation.concat_layer import concat_layer
from layers.hidden.modification.dropout_layer import dropout_layer
from layers.hidden.modification.flatten_layer import flatten_layer
from functions.activation.non_diff.relu import relu
from functions.activation.regularization.dropout import dropout
from functions.metric.accuracy import accuracy
from functions.metric.mean_squared_error import mean_squared_error
from functions.activation.smooth.softmax import softmax
from  layers.hidden.activation.rnn_layer.lstm_layer import lstm_layer
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
from TrainingSession.training_session import training_session
from TrainingStrategy.training_strategy import training_strategy
import nodes.handler
from functions.cost.cost_function import cost_function
from functions.loss.mse import mse
from functions.objective.MinimizeObjective.minimize_objective import minimize_objective
from layers.InOutLayer.InputLayer.input_layer import input_layer
from layers.InOutLayer.OutputLayer.output_layer import output_layer
from layers.InOutLayer.in_out_layer import in_out_layer

def handle_lstm( node,name):
    return lstm_layer(node,name)

def handle_rms_prop(node):
    return rms_prop(node)

def handle_gradient_descent( node):
    return gradient_descent(node)

def check_if_acc(node):
    children=[x for x in node.get_inputs()]
    for x in children:
        print(x.get_name())
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

def handle_dataset_pipe(network,nodes_list,type):
    dp_list=[]
    for i,_ in enumerate(nodes_list):
        helper = ""
        for elem in nodes.handler.entitiesHandler.node_map.keys():
            if nodes.handler.entitiesHandler.node_map[elem].search_inputs(nodes_list[i].name)==True:
                helper=nodes.handler.entitiesHandler.node_map[elem]
                break
        dp_list.append(dataset_pipe(network,nodes_list[i],helper,type,str(i)))
    return dp_list

def handle_random_normal(node):
    return
    '''
    name=get_inputs()[0].get_name()
    input_nm_1=handler.node_map[name].get_inputs()[0].get_name()
    name = get_inputs()[1].get_name()
    input_nm_2 = handler.node_map[name].get_inputs()[0].get_name()
    if handler.node_map[input_nm_1].get_op()=="Placeholder":
        print("2:", handler.node_map[input_nm_2].get_name(), " 1:", handler.node_map[input_nm_1].get_name())
        return accuracy(get_name(), "accuracy", handler.node_map[input_nm_1].get_name(),
                        handler.node_map[input_nm_2].get_name())
    else:
        tmp=handler.node_map[input_nm_1]
        while tmp.get_op() in handler.intermediate_operations:
            if len(tmp.get_inputs())!=1:
                import sys
                print("PROBLEM IN ACCURCY")
                sys.exit()
            tmp=handler.node_map[tmp.get_inputs()[0].get_name()]
        print("1:",handler.node_map[input_nm_2].get_name()," 2:",tmp.get_name())
        return accuracy(get_name(), "accuracy", handler.node_map[input_nm_2].get_name(),tmp.get_name())
    '''


def handle_softmax( node):
    return softmax(node.get_name(), node.node_obj)

def handle_adam(node,name):
    return adam(node,name)

def handle_sparse_cross_entropy(node,c_name):
    print(node.get_name())
    for dim in node.get_output():
        print(dim)
        for i,elem in enumerate(dim.dim):
            if elem.size != -1:
                if elem.size > 2:
                    return cost_function(c_name, categorical_cross_entropy(node.get_name(), node))
                else:
                    return cost_function(c_name, binary_cross_entropy(node.get_name(), node))

def handle_cross_entropy(node,name,c_name):
    for t_name in nodes.handler.entitiesHandler.node_map.keys():
        if "softmax_cross_entropy" in t_name:
            for elem in nodes.handler.entitiesHandler.node_map[t_name].get_inputs():
                if elem.get_op() == "Placeholder":
                    for dim in elem.get_attr()["shape"].shape.dim:
                        if dim.size != -1:
                            if dim.size > 2:
                                return cost_function(c_name,categorical_cross_entropy(name,node))
                            else:
                                return cost_function(c_name,binary_cross_entropy(name,node))
    # In case placeholder has only negative dimensions

def handle_flatten(node,name):
    return flatten_layer(node,name)

def handle_pow(node,network):
    children=node.get_inputs()
    sub=False
    power_of_two=False
    node_obj=None
    for elem in children:
        if elem.get_op()=="Sub":
            node_obj=elem
            sub=True
        if elem.get_op()=="Const":
            num=elem.node_obj.attr["value"].tensor.float_val
            print(int(num[0]))
            if int(num[0])==2:
                power_of_two=True
    if sub==True and power_of_two==True:
        return handle_mean_square_error(node_obj,network,node.get_name())
    else:
        return ""
def handle_neg_for_log(node):
    inputs=node.get_inputs()
    while inputs!=[]:
        tmp=[]
        for input in inputs:
            print("GKYRTIS=",input.get_name())
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
        print("GKYRTIS=", [x.get_name() for x in tmp])
        inputs=tmp

    return ""

def handle_log(node):
    return cost_function("_cost",categorical_cross_entropy(node.get_name(),node))



def handle_mean_square_error(node,network,name):
    '''
    for name in nodes.handler.entitiesHandler.node_map.keys():
        if nodes.handler.entitiesHandler.node_map[name].search_inputs(node.get_name())==True:
            print("This is not final mean.")
            return "
    '''
    #TODO:FIX FOR SIMPLE/SQUARE MEAN
    mean=mse(node,name)
    return cost_function(network+"_cost",mean)

def check_for_bias(name):
    nm = nodes.handler.entitiesHandler.node_map
    for node_name in nodes.handler.entitiesHandler.node_map.keys():
        out_name = nm[node_name].get_name()
        if nm[node_name].search_inputs(name) == True:
            if nm[node_name].get_op() == "BiasAdd":
                return out_name
            else:
                print("ERROR:Conv2d node with name ", name, "should have biasAdd following,this does not.")
                return -1

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

def handle_training_single(name,network,IOPipe,optimizer,epochs,batch):
    return training_single(name,network,IOPipe,optimizer,epochs,batch)

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

def handle_out_layer(node):
    return output_layer(node)

def handle_in_layer(node):
    return input_layer(node)

def handle_training_session(name,trStep,primaryTrStep):
    return training_session(name,trStep,primaryTrStep)


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
    #TODO:What happens if no bias in the node???
    print("HOELELELELELLELEL")
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
                        print("LOGGING:matMul with name ",matMul.get_name()," has less than 2 inpus.")
                        return (False,[])
                return (True, node)
    else:
        print("LOGGING:node ",node.get_name()," has less than 2 inpus.")
    return (False, [])