from generated_files_classes.file_tr_step import file_tr_step
from generated_files_classes.file_training_session import file_training_session
import types


def get_output_networks(sess_run):
    network_list = []
    if "sess.run" not in sess_run:
        optimizer=sess_run.split(".")[0].replace("-","")
        network_list.append(optimizer)
        return network_list
    #print("0=",sess_run)
    #print("1=",sess_run.split('sess.run('))
    if 'fetches=' in sess_run.replace(" ",""):
        sess_run=sess_run.replace("fetches=","")
    #print("2=",sess_run.split('sess.run(')[1].split(","))
    comma_separated=sess_run.replace(" ","").split('sess.run(')[1].split(",")
    complex=False
    for elem in comma_separated:
        if elem.count("[")!=elem.count("]"):
            complex=True
            if elem.count("[") > elem.count("]"):
                network_list.append(elem[1:])
            if elem.count("[") < elem.count("]"):
                network_list.append(elem[:-1])
    #if complex==True:
        #print(":Networks List=",network_list)
    if complex==False:
        networks = sess_run.split('sess.run(')[1].split(",")[0].replace(" ", "")
        network_list.append(networks.split(",")[0])
    return network_list

def get_feed_dict(sess_run):
    if "feed_dict=" not in sess_run.replace(" ",""):
        return sess_run.replace(" ", "").split(",")[-1].replace(")","")
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
    print("EPOCH IS RE ZOGRAFAKI ",epoch)
    sess_run=content.split("||||")[0]
    networks=get_output_networks(sess_run)
    feed_dict=get_feed_dict(sess_run)
    return (feed_dict,networks,epoch)

def handle_batch(content):
    batches_list=content.split("||||")
    return batches_list

def handle_network_var(value):
    #print("Value is ",value)
    if value.startswith("["):
        value=value.replace("[","").replace("]","").split(",")
        elem_lst={}
        elem_lst["L"] = []
        elem_lst["O"] = []
        for val in value:
            #print("Value is ", val)
            if "Tensor" in val:
                n_value = val.split("Tensor")[1].split("'")[1]
                elem_lst["L"].append(n_value)
            else:
                n_value = val.replace(" ", "").split("tf.Operation")[1].split("'")[1]
                elem_lst["O"].append(n_value)
        return ("B",elem_lst)
    if "Tensor" in value:
        n_value=value.split("Tensor(\"")[1].split("\"")[0]
        return ("L", n_value)
    else:
        if "name:" in value:
            n_value = value.replace(" ","").split("name:\"")[1].split("\"")[0]
        elif "tf.Operation" in value:
            n_value = value.replace(" ", "").split("tf.Operation")[1].split("'")[1]
        return ("O", n_value)

def handle_input_var(value,isDict=False,nVal=""):
    if isDict==True:
        value=value.replace(" ","")[1:-1]
        #print('list_val=',value)
        list_val=value.split(",")
        for elem in list_val:
            if elem.split(":")[0]==nVal:
                value=elem.split(":")[1]
                if "tf.Tensor" in value:
                    n_value = value.split("tf.Tensor")[1].replace(" ", "").split("'")[1].split(":")[0]
                else:
                    n_value = value.split("Tensor(\"")[1].split("\"")[0].split(":")[0]
                return n_value
    elif "tf.Tensor" in value:
        n_value=value.split("tf.Tensor")[1].replace(" ","").split("'")[1].split(":")[0]
    else:
        n_value = value.split("Tensor(\"")[1].split("\"")[0].split(":")[0]
    return n_value

def search_feed_dict(feed_dict,key,value,inputs):
    for input in feed_dict:
        outer_var=""
        if "[" in input and "]" in input:
            n_input=input.split("[")[1].split("]")[0]
            outer_var=input.split("[")[0]
        else:
            n_input=input
        #print("KEY=",key)
        if outer_var!="":
            if outer_var==key:
                in_ = handle_input_var(value,True,n_input)
                inputs.append(in_)
        elif n_input==key:
            in_ = handle_input_var(value)
            inputs.append(in_)
    return inputs

def handle_info(content,networks,feed_dict):
    dict=content.split("||||")
    loss=[]
    optimizer=[]
    inputs=[]
    new_networks=[]
    for network in networks:
        if "." in network:
            temp=network.split(".")
            new_networks.append(temp)
    for element in dict:
        value=""
        try:
            value=element.split("VALUE:")[1]
        except:
            continue
        key = element.split("VALUE:")[0].split("KEY:")[1]
        for network in networks:
            n_network=network
            if isinstance(network, list):
                import sys
                print("ERROR:Unable to handle objects that refer are class members")
                sys.exit()
            #print(":Network=",network)
            if n_network==key:
                #print("KEY=",key,"-",n_network)
                (case,n_value)=handle_network_var(value)
                if case=="L":
                    loss.append(n_value)
                elif case=="O":
                    if "/" in n_value:
                        n_value=n_value.split("/")[-1]
                    optimizer.append(n_value)
                else:
                    for key in n_value.keys():
                        for elem in n_value[key]:
                            if key == "L":
                                loss.append(elem)
                            else:
                                optimizer.append(elem)
                break
        if str(type(feed_dict))=="<class 'list'>":
            inputs=search_feed_dict(feed_dict,key,value,inputs)
        else:
            if feed_dict in key:
                vars=value.split("<tf.Tensor ")
                vars=vars[1:]
                for var in vars:
                    inputs.append(var.split("'")[1].split(":")[0])
    #print("Returning inputs=",inputs)
    return(loss,optimizer,inputs)


def handle_lines_and_info(files,pathlistInfo,pathlistLine,pathlistBatch,pathlistSession):
    pathlistInfo=list(pathlistInfo)
    pathlistLine = list(pathlistLine)
    pathlistBatch=list(pathlistBatch)
    pathlistSession=list(pathlistSession)
    total_sessions_dict={}
    for session in pathlistSession:
        session_name=str(session).split("]_")[0]+"]"
        with open(str(session), 'r') as content_file:
            content = content_file.read()
            session_epochs=content.count("----")
        print("LOGGING:Session name is : ",session_name," Epochs :",session_epochs)
        fts=file_training_session(session_name,session_epochs,[])
        for file in pathlistLine:
            print("LOGGING:Begin searching for ",file)
            if  session_name in str(file) :
                #print("Found line-file : ",file)
                with open(str(file), 'r') as content_file:
                    content = content_file.read()
                    (feed_dict,networks,epoch)=handle_lines(content)
                    print("LOGGING:-----------------------------------------------")
                    print("LOGGING:Found information from handling produced files")
                    print("LOGGING:Feed Dict:",feed_dict)
                    print("LOGGING:Networks:",networks)
                    print("LOGGING:Epoch:", epoch)
                    print("LOGGING:-----------------------------------------------")
                new_file_training=file_tr_step()
                step_name=str(file).replace(".lines","")
                new_file_training.name=str(step_name)
                for file_ in pathlistInfo:
                    if str(file).replace(".lines","")==str(file_).replace(".info",""):
                        with open(str(file_), 'r') as content_file:
                            content = content_file.read()
                            (loss,optimizer,inputs)=handle_info(content,networks,feed_dict)
                for file_ in pathlistBatch:
                    if str(file).replace(".lines", "") == str(file_).replace(".batch", ""):
                        with open(str(file_), 'r') as content_file:
                            content = content_file.read()
                            (batch_list)=handle_batch(content)
                            new_file_training.batches=batch_list
                            new_file_training.loss = loss
                            new_file_training.optimizer = optimizer
                            new_file_training.epoch = epoch
                            new_file_training.inputs = inputs
                            fts.steps.append(new_file_training)
        total_sessions_dict[fts.name]=fts
    for key in total_sessions_dict.keys():
        print("\n\nLOGGING:Session information along with steps:")
        total_sessions_dict[fts.name].print()
        print("LOGGING:End session information along with steps.\n\n")
    return total_sessions_dict


def find_next_session_and_step(sessions,timeList):
    new_sessions=sessions.copy()
    for sess in sessions.keys():
        new_step_list = []
        if len(sessions[sess].steps)>1:
            counter=0
            list_index = []
            for step in sessions[sess].steps:
                list_index.append([step,timeList.index(step.name)])
            list_index = sorted(list_index, key=lambda x: x[1])
            while counter<len(list_index)-1:
                list_index[counter][0].next =list_index[counter+1][0].name
                new_step_list.append(list_index[counter][0])
                counter+=1
            new_step_list.append(list_index[-1][0])
        else:
            new_step_list=sessions[sess].steps
        new_sessions[sess].steps=new_step_list
    list_sessions_index=[]
    if len(sessions.keys())>1:
        for sess in sessions.keys():
            if len(sessions[sess].steps) > 0:
                list_sessions_index.append([sessions[sess], timeList.index(sessions[sess].steps[0].name)])
            else:
                print("EXTREME ERROR FOR SESSION-NO STEP=",sessions[sess].name)
                import sys
                sys.exit()
        list_sessions_index = sorted(list_sessions_index, key=lambda x: x[1])
        counter=0
        while counter < len(list_sessions_index) - 1:
            list_sessions_index[counter][0].next_session = list_sessions_index[counter + 1][0].name
            new_sessions[list_sessions_index[counter][0].name]=list_sessions_index[counter][0]
            counter+=1
    return new_sessions


def update_inner_epochs(sessions):
    ret_sessions=sessions.copy()
    for sess in sessions.keys():
        co_train=""
        found_co_train = False
        for step in sessions[sess].steps:
            if co_train in step.name:
                found_co_train=True
                break
        if found_co_train==True:
            new_steps=divide_epochs(sessions[sess].steps,sessions[sess].session_epoch)
            ret_sessions[sess].steps=new_steps
    return ret_sessions


def divide_epochs(steps,sess_epochs):
    new_steps=[]
    is_co_trained=False
    for step in steps:
        if "co_train" in step.name:
            is_co_trained=True
            break
    if is_co_trained==True:
        for step in steps:
            new_step=step
            if step.epoch!=sess_epochs:
                print("ZOGRAFAKIMOUNARA=",step.name,"-step-",step.epoch,"-sess-",sess_epochs)
                new_step.epoch=int(new_step.epoch/sess_epochs)
            else:
                print("ZOGRAFAKIPORNH=",step.name,"-step-",step.epoch,"-sess-",sess_epochs)
            new_steps.append(new_step)
        return new_steps
    return steps
