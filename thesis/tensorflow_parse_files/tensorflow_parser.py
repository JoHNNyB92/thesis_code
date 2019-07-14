from nodes.nodeEntity import nodeEntity as nodeClass
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler
from handle_entities import handle_entities
import print_info

def put_in_map(node):
    inputs=[]
    nodes.handler.entitiesHandler.node_map[node.name] = nodeClass(node, inputs)

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

def parse_pbtxt(path,part_name):
    graph_def = graph_pb2.GraphDef()
    with open(path, "rb") as f:
        text_format.Merge(f.read(), graph_def)
    for node in graph_def.node:
        put_in_map(node)
    for node in graph_def.node:
        put_inputs_in_map(node)
    #TODO: PER NETWORK THIS SHOULD BE DONE
    curr=nodes.handler.entitiesHandler.current_network
    for e in nodes.handler.entitiesHandler.node_map.keys():
        nodes.handler.entitiesHandler.handle_different_layer_cases(e)
    nodes.handler.entitiesHandler.clear_layers()
    inputs=[]
    outputs=[]
    for elem in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        print("=============================================")
        print("Finding input for layer=",elem)
        nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].find_input_layer()
        print("=============================================")
    for elem in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        #If only one input and that input is placeholder then it an input layer
        if nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].is_input==True and \
            len(nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].previous_layer)==1:
            inputs.append(elem)
    #nodes.handler.entitiesHandler.transform_layers_to_ins_outs(inputs)
    result=print_info.print_topology()
    if result!="success":
        print("PARSING FAILED")
        return result
    #TODO FOR MULTIPLE NETWORKS WE NEED MULTIPLE EVALUATION RESULTS->THUS THIS SHOULD BE MOVED
    '''
    res=nodes.handler.entitiesHandler.check_multiple_networks()
    if res==0:
        nodes.handler.entitiesHandler.prepare_strategy(part_name)
    elif res==-1:
        print("ERROR:Program not a network finally")
        return "ERROR:This tensorflow program is not a network.No objective functions identified"
    print("-----------------------------------------------------")
    '''
    return result

def begin_parsing(name,pbtxt_file,log_file):
    nodes.handler.entitiesHandler=handle_entities()
    part_name=name.replace(".py","")
    rdfWrapper.log_file=log_file
    rdfWrapper.new_init_data(part_name)
    rdfWrapper.new_init_new_network(part_name)
    rdfWrapper.new_init_new_evaluation(part_name+"_eval",part_name)
    #nodes.handler.entitiesHandler.set_batch_epoch(batch,epoch)
    result=parse_pbtxt(pbtxt_file,part_name)
    #nodes.handler.entitiesHandler=""
    return (result,nodes.handler.entitiesHandler)

def insert_in_annetto():
    rdfWrapper.insert_ann_graph()
    nodes.handler.entitiesHandler=""



