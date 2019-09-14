from nodes.nodeEntity import nodeEntity as nodeClass
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler
from handle_entities import handle_entities
import print_info

#NodeClass object contains the pbtxt nodes produced by the tensorflow program.
def put_in_map(node):
    inputs=[]
    nodes.handler.entitiesHandler.node_map[node.name] = nodeClass(node, inputs)

#Create a map to store the nodeClass objects identified from pbtxt file.
def put_inputs_in_map(node):
    inputs = []
    for elem in node.input:
        rest=elem.replace("^", "")
        tmp2=rest.split("/")
        if len(tmp2)!=0:
            tmp3=tmp2[len(tmp2)-1]
            if ":" in tmp3:
                rest = rest.split(":", 1)[0]
        rest=rest.replace("^","")
        inputs.append(nodes.handler.entitiesHandler.node_map[rest])
    nodes.handler.entitiesHandler.node_map[node.name].inputs=inputs

#Parse file and iterate through all nodes produced that belong to the neural network,call function to store them
#in a map.It also call function to handle the node based on the respective case.
def parse_pbtxt(path):
    #Protocol file parse and read.
    graph_def = graph_pb2.GraphDef()
    with open(path, "rb") as f:
        text_format.Merge(f.read(), graph_def)
    for node in graph_def.node:
        put_in_map(node)
    for node in graph_def.node:
        put_inputs_in_map(node)
    curr=nodes.handler.entitiesHandler.current_network
    #Iterate through all nodes.
    for e in nodes.handler.entitiesHandler.node_map.keys():
        nodes.handler.entitiesHandler.handle_different_layer_cases(e)
    #Some layer can be part of a bigger layer,thus those layers should be removed from in here.
    nodes.handler.entitiesHandler.clear_layers()
    #For each layer identified,find input and output layers.
    for elem in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        print("=============================================")
        print("Finding input for layer=",elem)
        nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].find_input_layer()
        print("=============================================")
    result=print_info.print_topology()
    if result!="success":
        print("LOGGING:Parsing failed.")
    return result

#Initial call for parsing pbtxt
def begin_parsing(name,pbtxt_file,log_file,counter):
    nodes.handler.entitiesHandler=handle_entities()
    part_name=name.replace(".py","")
    rdfWrapper.log_file=log_file
    #Initalize some basic information for the neural network.
    #The data class,storing information for inserting into annetto
    rdfWrapper.new_init_data(part_name,counter)
    #New network name
    rdfWrapper.new_init_new_network(part_name)
    #New evaluation function
    rdfWrapper.new_init_new_evaluation(part_name+"_eval",part_name)
    result=parse_pbtxt(pbtxt_file)
    return (result,nodes.handler.entitiesHandler)

#Function to insert into the annetto triple store database
def insert_in_annetto():
    cnt=rdfWrapper.insert_ann_graph()
    #Reset variables used for different names on multiple networks inserted.
    rdfWrapper.reset_vars()
    #Reset also entitieshandler class
    nodes.handler.entitiesHandler=""
    return cnt



