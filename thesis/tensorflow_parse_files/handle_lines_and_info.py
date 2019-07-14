from file_training import file_training

def get_output_networks(sess_run):
    networks=sess_run.split('sess.run(')[1].split(",")[0].replace(" ","")
    network_list = []
    if "[" in networks:
        if "," in networks:
            print('1')
            for elem in networks.split(','):
                network_list.append(elem.replace(" ",""))
        else:
            print("2")
            network_list.append(networks.replace(" ",""))
        #print("networks are ",network_list)
    else:
        network_list.append(networks)
    return network_list

def get_feed_dict(sess_run):
    feed_dict = sess_run.replace(" ","").split("feed_dict=")[1]
    feed=[]
    if "{" in feed_dict:
        feed_dict=feed_dict.split("{")[1].split("}")[0]
        if "," in feed_dict:
            for elem in feed_dict.split(','):
                feed.append(elem.replace(" ","").split(":")[0])
        else:
            print("2=",feed_dict.replace(" ","").split(":")[0])
            feed.append(feed_dict.replace(" ","").split(":")[0])
    else:
        feed_dict = feed_dict.split(")")[0]
        feed=feed_dict
    #print("feed_dict is ",feed)
    return feed


def handle_lines(content):
    epoch=content.count("----")
    #print("Epoch is ",epoch)
    sess_run=content.split("||||")[0]
    #print("sess run is =",sess_run)
    networks=get_output_networks(sess_run)
    feed_dict=get_feed_dict(sess_run)
    return (feed_dict,networks,epoch)


def handle_network_var(value):
    if "Tensor" in value:
        n_value=value.split("Tensor(\"")[1].split("\"")[0]
        return ("L", n_value)
    else:
        print(value)
        n_value = value.replace(" ","").split("name:\"")[1].split("\"")[0]
        return ("O", n_value)

def handle_input_var(value):
    if "tf.Tensor" in value:
        n_value=value.split("tf.Tensor")[1].replace(" ","").split("'")[1].split(":")[0]
    else:
        print("MIASMA=",value,"-")
        n_value = value.split("Tensor(\"")[1].split("\"")[0].split(":")[0]
    return n_value

def search_feed_dict(feed_dict,key,value,inputs):
    for input in feed_dict:
        if input==key:
            print("KEY=",key,"-",input)
            in_ = handle_input_var(value)
            inputs.append(in_)
    return inputs

def handle_info(content,networks,feed_dict):
    dict=content.split("||||")
    loss=[]
    optimizer=[]
    inputs=[]
    input_search=[]
    not_class=False
    for element in dict:
        value=""
        try:
            value=element.split("VALUE:")[1]
        except:
            continue
        key = element.split("VALUE:")[0].split("KEY:")[1]
        for network in networks:

            if network in key:
                (case,n_value)=handle_network_var(value)
                if case=="L":
                    loss.append(n_value)
                if case=="O":
                    optimizer.append(n_value)
                break
        #print("feed=",feed_dict)
        if str(type(feed_dict))=="<class 'list'>":
            inputs=search_feed_dict(feed_dict,key,value,inputs)
        else:
            #not_class=True
            if feed_dict in key:
                #print("ARETH-1=",key)
                #print("ARETH-0.5=", value)
                vars=value.split("<tf.Tensor ")
                vars=vars[1:]
                #print("AReth0=",vars)
                for var in vars:
                    #print("Areth1=",var.split("'"))
                    #print("Areth3=",var.split("'")[1].split(":")[0])
                    inputs.append(var.split("'")[1].split(":")[0])
    '''
    if not_class==True:
        for element in dict:
            value = ""
            try:
                value = element.split("VALUE:")[1]
            except:
                continue
            key = element.split("VALUE:")[0].split("KEY:")[1]
            print("EVITA=",input_search,"key=",key)
            inputs=search_feed_dict(input_search,key,value,inputs)
            print("ARETH2=", inputs)
    '''
    print("Returning inputs=",inputs)
    return(loss,optimizer,inputs)

def handle_lines_and_info(files,pathlistInfo,pathlistLine):
    pathlistInfo=list(pathlistInfo)
    file_training_information={}
    for file in pathlistLine:
        print("begin searching for ",file)
        found_file = False
        for prFile in files:
            print("File=",prFile)
            if str(prFile)+".lines" in str(file):
                print("\n\n\n\n\n\n\n\nFound str(prFile)=",str(prFile))
                print('file=',file)
                with open(str(file), 'r') as content_file:
                    content = content_file.read()
                    (feed_dict,networks,epoch)=handle_lines(content)
                #print("Networks=",networks)
                new_file_training=file_training()
                new_file_training.name=str(prFile)
                for file_ in pathlistInfo:
                    if (str(prFile)+".info") in str(file_):
                        with open(str(file_), 'r') as content_file:
                            content = content_file.read()
                            print("NETWORKS=",networks)
                            (loss,optimizer,inputs)=handle_info(content,networks,feed_dict)
                            new_file_training.loss=loss
                            new_file_training.optimizer=optimizer
                            new_file_training.epoch=epoch
                            new_file_training.inputs = inputs
                            new_file_training.print()
                            file_training_information[new_file_training.name]=new_file_training
                            found_file=True
                            break
                    if found_file==True:
                        break
                if found_file==True:
                    break

    return file_training_information