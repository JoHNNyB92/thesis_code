from file_training import file_training

def get_output_networks(sess_run):
    networks=sess_run.split('[')[1].split("]")[0].replace(" ","")
    network_list=[]
    if "," in networks:
        print('1')
        for elem in networks.split(','):
            network_list.append(elem.replace(" ",""))
    else:
        print("2")
        network_list.append(networks.replace(" ",""))
    print("networks are ",network_list)
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
        feed.append(feed_dict)
    print("feed_dict is ",feed)
    return feed


def handle_lines(content):
    epoch=content.count("||||")
    print("Epoch is ",epoch)
    sess_run=content.split("||||")[0]
    print("sess run is =",sess_run)
    networks=get_output_networks(sess_run)
    feed_dict=get_feed_dict(sess_run)
    return (feed_dict,networks,epoch)


def handle_network_var(value):
    if "Tensor" in value:
        n_value=value.split("Tensor(\"")[1].split("\"")[0]
        return ("L", n_value)
    else:
        n_value = value.replace(" ","").split("name:\"")[1].split("\"")[0]
        return ("O", n_value)

def handle_input_var(value):
    if "tf.Tensor" in value:
        n_value=value.split("tf.Tensor")[1].replace(" ","").split("'")[1].split(":")[0]
    else:
        n_value = value.split("Tensor(\"")[1].split("\"")[0].split(":")[0]
    return n_value

def handle_info(content,networks,feed_dict):
    dict=content.split("||||")
    loss=[]
    optimizer=[]
    inputs=[]
    for element in dict:
        value=""
        try:
            value=element.split("VALUE:")[1]
        except:
            continue
        key = element.split("VALUE:")[0].split("KEY:")[1]
        for network in networks:
            #print("Search for ",network)
            if network in key:
                print("KEY IS ",key,"Value:",value)
                (case,n_value)=handle_network_var(value)
                if case=="L":
                    loss.append(n_value)
                if case=="O":
                    optimizer.append(n_value)
                break
        print("feed=",feed_dict)
        if str(type(feed_dict))=="<class 'list'>":
          for input in feed_dict:
              if input in key:
                  print("VAlu=",value)
                  in_=handle_input_var(value)
                  print("ALELELELELELEL=",in_)
                  inputs.append(in_)

    return(loss,optimizer,inputs)

def handle_lines_and_info(files,pathlistInfo,pathlistLine):
    print("\n\n\n\n\n\n\n\n\n\n\n\n")
    pathlistInfo=list(pathlistInfo)
    for file in pathlistLine:
        for prFile in files:
            if str(prFile)+".lines" in str(file):
                with open(str(file), 'r') as content_file:
                    content = content_file.read()
                    (feed_dict,networks,epoch)=handle_lines(content)
                print("Networks=",networks)
                new_file_training=file_training()
                new_file_training.name=str(prFile)+".lines"
                for file_ in pathlistInfo:
                    for prFile_ in files:
                        if str(prFile_)+".info" in str(file_):
                            with open(str(file_), 'r') as content_file:
                                content = content_file.read()
                                (loss,optimizer,inputs)=handle_info(content,networks,feed_dict)
                                new_file_training.loss=loss
                                new_file_training.optimizer=optimizer
                                new_file_training.epoch=epoch
                                new_file_training.inputs = inputs
                                new_file_training.print()
                                break

