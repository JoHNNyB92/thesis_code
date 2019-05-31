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

def parse_json(path,epoch,batch,part_name):
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
    for elem in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        print("=============================================")
        print("Finding input for layer=",elem)
        nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].find_input_layer()
        print("=============================================")
        print("RESULT2=",elem,"-",nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[elem].previous_layer)
    print_info.print_topology()
    import sys
    sys.exit()
    #TODO FOR MULTIPLE NETWORKS WE NEED MULTIPLE EVALUATION RESULTS->THUS THIS SHOULD BE MOVED
    #print_info.print_topology()
    print("=============================================================")

    if nodes.handler.entitiesHandler.check_multiple_networks()!=0:
        nodes.handler.entitiesHandler.prepare_strategy(batch, epoch, part_name)
        nodes.handler.entitiesHandler.insert_to_evaluation_pipe()
        print("=============================================================")
    import sys
    sys.exit()


    #nodes.handler.entitiesHandler.check_if_ae()
    print("Starting inserting to annetto.")
    print("-----------------------------------------------------")
    rdfWrapper.insert_ann_graph()

    #print_info.print_topology()
    print("Finished inserting to annetto for file :",path,"\n\n\n\n\n\n\n\n")

def begin_parsing(name,pbtxt_file,epoch,batch):
    nodes.handler.entitiesHandler=handle_entities()
    part_name=name.replace(".py","")
    print("part_name=",part_name)
    rdfWrapper.new_init_data(part_name)
    rdfWrapper.new_init_new_network(part_name+"_net")
    rdfWrapper.new_init_new_evaluation(part_name+"_eval",part_name+"_net")
    nodes.handler.entitiesHandler.set_batch_epoch(batch,epoch)
    parse_json(pbtxt_file,epoch,batch,part_name)
    nodes.handler.entitiesHandler=""
