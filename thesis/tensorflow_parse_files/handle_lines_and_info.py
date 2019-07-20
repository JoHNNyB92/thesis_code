from generated_files_classes.file_tr_step import file_tr_step
from generated_files_classes.file_training_session import file_training_session

def get_output_networks(sess_run):
    print("0=",sess_run)
    print("1=",sess_run.split('sess.run('))
    print("2=",sess_run.split('sess.run(')[1].split(","))
    print("3=",sess_run.split('sess.run(')[1].split(",")[0].replace(" ",""))
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
            feed.append(feed_dict.replace(" ","").split(":")[0])
    else:
        feed_dict = feed_dict.split(")")[0]
        feed=feed_dict
    return feed


def handle_lines(content):
    epoch=content.count("----")
    sess_run=content.split("||||")[0]
    networks=get_output_networks(sess_run)
    feed_dict=get_feed_dict(sess_run)
    return (feed_dict,networks,epoch)

def handle_batch(content):
    batches_list=content.split("||||")
    return batches_list

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
        n_value = value.split("Tensor(\"")[1].split("\"")[0].split(":")[0]
    return n_value

def search_feed_dict(feed_dict,key,value,inputs):
    for input in feed_dict:
        if input==key:
            in_ = handle_input_var(value)
            inputs.append(in_)
    return inputs

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

            if network in key:
                (case,n_value)=handle_network_var(value)
                if case=="L":
                    loss.append(n_value)
                if case=="O":
                    optimizer.append(n_value)
                break
        if str(type(feed_dict))=="<class 'list'>":
            inputs=search_feed_dict(feed_dict,key,value,inputs)
        else:
            if feed_dict in key:
                vars=value.split("<tf.Tensor ")
                vars=vars[1:]
                for var in vars:
                    inputs.append(var.split("'")[1].split(":")[0])
    print("Returning inputs=",inputs)
    return(loss,optimizer,inputs)


def handle_lines_and_info(files,pathlistInfo,pathlistLine,pathlistBatch,pathlistSession):
    pathlistInfo=list(pathlistInfo)
    pathlistLine = list(pathlistLine)
    pathlistBatch=list(pathlistBatch)
    pathlistSession=list(pathlistSession)
    new_file_training=""
    total_sessions_dict={}
    for session in pathlistSession:
        session_name=str(session).split("]_")[0]
        with open(str(session), 'r') as content_file:
            content = content_file.read()
            session_epochs=content.count("----")
        fts=file_training_session(session_name,session_epochs,[])
        for file in pathlistLine:
            print("begin searching for ",file)
            found_file = False
            print("Session=",session_name+"]")
            if  session_name+"]" in str(file) :
                print("\n\n\nFound line file=")
                print('file=',file)
                with open(str(file), 'r') as content_file:
                    content = content_file.read()
                    (feed_dict,networks,epoch)=handle_lines(content)
                new_file_training=file_tr_step()
                step_name=str(file).replace(".lines","")
                new_file_training.name=str(step_name)
                for file_ in pathlistInfo:
                    if str(file).replace(".lines","")==str(file_).replace(".info",""):
                        with open(str(file_), 'r') as content_file:
                            content = content_file.read()
                            print("NETWORKS=",networks)
                            (loss,optimizer,inputs)=handle_info(content,networks,feed_dict)
                            new_file_training.loss=loss
                            new_file_training.optimizer=optimizer
                            new_file_training.epoch=epoch
                            new_file_training.inputs = inputs
                for file_ in pathlistBatch:
                    print("file=",file," file_=",file_)
                    if str(file).replace(".lines", "") == str(file_).replace(".batch", ""):
                        with open(str(file_), 'r') as content_file:
                            content = content_file.read()
                            (batch_list)=handle_batch(content)
                            new_file_training.batches=batch_list
                            new_file_training.print()
                            found_file=True
                            fts.steps.append(new_file_training)
                            print("\n\n\n\n added to ",session_name," \n",new_file_training.name,"\nsize ",len(fts.steps))
                            break
                    if found_file==True:
                        break
        total_sessions_dict[fts.name]=fts
    return total_sessions_dict