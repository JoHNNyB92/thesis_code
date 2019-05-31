import nodes.handler
def print_topology():
    curr = nodes.handler.entitiesHandler.current_network
    print("\n Layers \n")
    for key in nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer.keys():
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].name)
        print("NL:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].next_layer)
        print("AC:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].activation)
        print("PL:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].previous_layer)
        print("OUT:", nodes.handler.entitiesHandler.data.annConfiguration.networks[curr].layer[key].output_nodes)
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
        print("\n evaluation \n")
        print("_____________________________________________________________________________________")
        print("NM:", nodes.handler.entitiesHandler.data.evaluationResult.name)
        print("CF:", nodes.handler.entitiesHandler.data.evaluationResult.metric.name)
        print("_____________________________________________________________________________________")
    else:
        print("\nno evaluation\n")

