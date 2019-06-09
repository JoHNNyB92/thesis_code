import nodes.handler
def print_topology():
    curr = nodes.handler.entitiesHandler.current_network
    if len(nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys())==0:
        print("ERROR:Program not a network finally")
        return "ERROR:This tensorflow program is not a network.0 layers were indentified."

    print("\n Layers \n")
    del_layers=[]
    for key in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].name)
        print("NL:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].next_layer)
        if nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].activation!=None:
            print("AC:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].activation.name)
        else:
            print("AC:",None)
        print("PL:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].previous_layer)
        print("OUT:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].output_nodes)
        if nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].previous_layer==[] and nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].next_layer==[] \
            and nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].node.get_op()!="Placeholder":
            print("LAYER TO BE DELETED,IT IS WITHOUT INPUT OUTPUT LAYER")
            del_layers.append(key)

        for layer in del_layers:
            del nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[layer]

        print("_____________________________________________________________________________________")
    print("\n Optimization \n")
    for key in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].optimizer.keys():
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].optimizer[key].name)
        print("TP:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].optimizer[key].type)
        print("LR:",  nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].optimizer[key].learning_rate)
        print("_____________________________________________________________________________________")

    print("\n Objective \n")
    for key in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].objective.keys():
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].objective[key].name)
        print("CF:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].objective[key].cost_function.name)
        print("LO:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].objective[key].cost_function.loss.name)
        print("_____________________________________________________________________________________")
    if nodes.handler.entitiesHandler.data.evaluationResult!="":
        print("\n Evaluation \n")
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.evaluationResult.name)
        if nodes.handler.entitiesHandler.data.evaluationResult.metric!="":
            print("CF:", nodes.handler.entitiesHandler.data.evaluationResult.metric.name)
        print("_____________________________________________________________________________________")
    else:
        print("\nno evaluation\n")
    return ""
