import re
from shutil import copy
import shlex, subprocess
import os
import ntpath
import sys
import github.github as github
from sklearn.utils.bench import total_seconds


def execute_file(new_file,new_line_list,path_to_folder):
    try:
        os.remove(new_file)
    except OSError:
        pass
    with open(new_file, 'w',errors="replace") as f:
        for line in new_line_list:
            f.write(line)
    file_to_execute = ntpath.basename(new_file)
    os.chdir(path_to_folder)
    print("LOGGING:Executing file ", file_to_execute)
    try:
        subprocess.check_output([sys.executable, file_to_execute], stderr=subprocess.STDOUT)
        return file_to_execute
    except subprocess.CalledProcessError as e:
        return "error"

def handle_imported_files(proj):
    print("Started searching for project included files")
    produced_files=[]
    for path in proj:
        line_list=[]
        for line in open(path, errors="replace"):
            line_list.append(line)
        dir_ = os.getcwd()
        os.chdir(os.path.dirname(str(path)))
        (new_line_list,found,produced_files_one_file)=find_epoch_size(line_list,str(path))
        for produced_file in produced_files_one_file:
            produced_files.append(produced_file)
        import ntpath
        real_path=ntpath.basename(str(path))
        if found==True:
            f=open(real_path,'w')
            print("Writing to file ",real_path," get=",os.getcwd())
            for elem in new_line_list:
                f.write(elem)
            f.close()
        os.chdir(str(dir_))
    return produced_files

def parse_file(path,tf_run_app,proj):
    path_to_folder=os.path.dirname(path)
    file=ntpath.basename(path)
    new_file=path_to_folder+"/"+file
    (no_execution,line_list,produced_files)=check_if_already_executed(path,proj)
    if no_execution==True:
        return ("no execution",list(set(produced_files)))
    new_line_list=create_new_file(line_list,path,file,tf_run_app)
    dir_ = os.getcwd()
    os.chdir(os.path.dirname(str(path)))
    (new_line_list,_,produced_files)=find_epoch_size(new_line_list,path)
    os.chdir(str(dir_))
    handle_imported_files(proj)
    result=execute_file(new_file,new_line_list,path_to_folder)
    if result=="error":
        print("ERROR:File contains inner error,cannot execute it.")
        return ("error",[])
    else:
        print("LOGGING:Successfully executed.")
        return ("success",produced_files)


def check_if_already_executed(path,proj):
    prod_files=[]
    line_list=[]
    no_execution=False
    for ind, line in enumerate(open(path, errors="replace")):
        if "_sEssIOn_" in line:
            print("LINE=", line)
            print("LINE=", line.split('\'')[1])
            prod_files.append(line.split('\'')[1].split('.')[0])
            no_execution = True
        line_list.append(line)
    for filepath in proj:
        for ind, line in enumerate(open(filepath, errors="replace")):
            if "_sEssIOn_" in line:
                print("LINE=", line)
                print("LINE=", line.split('\'')[1])
                prod_files.append(line.split('\'')[1].split('.')[0])
                no_execution = True

    return (no_execution,line_list,prod_files)



def find_epoch_size(line_list,file_path):
    new_line_list = []
    cnt=0
    found_model_line=0
    num_of_space=0
    found_sess=False
    line_of_sess_run=""
    file=file_path.split("\\")[2:]
    file="_".join(file)
    found_run=False
    produced_files=[]
    session_counter=0
    added_lines=0
    before_sess_run=0
    files_added=0
    found_total_session=False
    first_for_space=-1
    close_=""
    last_line=False
    for ind_,line in enumerate(line_list):
        new_line_list.append(line)
        if line.isspace() == True:
            continue
        if ind_ == len(line_list) - 1:
            last_line = True
        line_space=len(line) - len(line.lstrip(' '))
        print("Gound total ses=",found_total_session,"-",line)
        if found_total_session==True:
            print("DEBUG=1")
            print("first_for_space=",first_for_space)
            print("line_space=", line_space)
            print("Line=",line)
            if first_for_space>=line_space or ind_==len(line_list)-1:
                print("DEBUG:Epoch size:New for found  ", line)
                found_total_session = False
                added_lines+=1
                print("ARXIDOMOURIS=",line)
                if last_line==True:
                    print("ALEXI")
                    new_line_list.append("\n"+close_)
                else:
                    print("ALEXI2")
                    new_line_list=new_line_list[:-1]+[close_]+new_line_list[-1:]
        #num_of_space = len(line) - len(line.lstrip(' '))
        if ".run" in line:
            num_of_space = len(line) - len(line.lstrip(' '))
            print("Found run=",num_of_space)
            line_of_sess_run=line
            found_run=True
            added_lines = len(new_line_list) - ind_ - 1
            before_sess_run=ind_+added_lines
            print("1BEFORE ADDING=", new_line_list[before_sess_run])
        elif found_run==True:
            print("Continuous line for sess run")
            line_of_sess_run+=line
        if "feed_dict" in line:
            found_model_line = 1
        if found_model_line == 1:
            if line.replace("\n", "").endswith(')') == True:
                line_of_sess_run=line_of_sess_run.replace("\n", "").replace(" ", "")
                line_list_for_feed = handle_feed_dict(line_of_sess_run, num_of_space)
                if ind_+1 >=len(line_list) or line_list[ind_ + 1][0].replace(" ","") != ".":
                    added_lines = len(new_line_list) - ind_-1
                    temp_ind = ind_ + added_lines
                    prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                    is_co_train=False
                    session_counter+=1
                    for_counter=0
                    write_ind=0
                    write_space=0
                    if found_total_session==False:
                        print("OLD ARXIDOMOURIS2=",line," new line list=",new_line_list[temp_ind])
                        (number_of_fors,session_for_ind,first_for_space,session_fors)=find_number_of_fors(temp_ind,new_line_list)
                    while end_loop(for_counter,new_line_list[temp_ind],number_of_fors,found_total_session) ==False:
                        if "sess.run" in new_line_list[temp_ind] and num_of_space!=prev_line_space:
                            is_co_train=True
                        if "_sEssIOn_" in new_line_list[temp_ind]:
                            files_replace=new_line_list[temp_ind].split('\'')[1].split('.')[0]
                            print("FOUND SESSION WITH REGEXP=",new_line_list[temp_ind])
                            reg = r'\[[\s\S]*\]'
                            if files_replace not in files_replace:
                                produced_files.remove(files_replace)
                                files_replace.append(files_replace)
                                produced_files.append(new_line_list[temp_ind].split('\'')[1].split('.')[0])
                            print("re file=", files_replace, " ", new_line_list[temp_ind])
                            new_line_list[temp_ind]=re.sub(reg, '['+str(session_counter)+']', new_line_list[temp_ind])
                        if new_line_list[temp_ind].replace(" ", "").startswith("for"):
                            print("3ELENHLINE=",new_line_list[temp_ind]+"---",for_counter)
                            for_counter += 1
                            if num_of_space > prev_line_space and for_counter==1:
                                write_ind = temp_ind
                                print("POUTANES=",new_line_list[temp_ind])
                                write_space = len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                                print("WRITE SPACE =", write_space)
                        temp_ind-=1
                        prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                    if found_total_session==False:
                        found_total_session=True
                        total_session = "_temporary_" + file.replace(".py","") + "_" + "_sEssIOn_[" + str(
                            session_counter) + "]_" + str(
                            session_counter) + ".total_session"
                        temp_space= len(new_line_list[session_for_ind+1]) - len(new_line_list[session_for_ind+1].lstrip(' '))
                        open_=(first_for_space)* " " + "total_session_abc = open('" + total_session + "', 'w')\n"
                        write_=(temp_space)* " " + "total_session_abc.write('----')\n"
                        close_=(first_for_space)* " " + "total_session_abc.close()\n"
                        new_line_list = new_line_list[:session_for_ind] + [open_] + new_line_list[session_for_ind:]
                        new_line_list = new_line_list[:session_for_ind +2] + [write_] + new_line_list[
                                                                                        session_for_ind +2:]
                        if session_fors%2==1:
                            write_ind+=1
                        else:
                            write_ind += 2
                        before_sess_run+=5
                        #print("POUSTH2=",new_line_list[before_sess_run])
                    else:
                        print("FOUND TOTAL SESSION WITH REGEXP=", new_line_list[temp_ind])
                        reg = r'\[[\s\S]*\]'
                        new_line_list[temp_ind] = re.sub(reg, '[' + str(session_counter) + ']',
                                                         new_line_list[temp_ind])
                        print("NEW LINE IS FOUND TOTAL SESSION WITH REGEXP=", new_line_list[temp_ind])
                        before_sess_run += 3
                    files_added+=1
                    (write_file, write_file_line, write_file_batch,temp_list, name)=prepare_lists_and_lines(is_co_train,file,session_counter,write_space,file_path)
                    #print("POUSTH4=", new_line_list[before_sess_run])
                    (prod_file,new_line_list)=append_file_lines(name,line_of_sess_run, \
                                      new_line_list,temp_list,line_list_for_feed, \
                                      write_file,write_file_line,write_file_batch,write_ind,
                                      before_sess_run,num_of_space,files_added)
                    print("POUSTH5=", new_line_list[before_sess_run])
                    print("PRODUCED FILE ADDED ",prod_file)
                    produced_files.append(prod_file)
                    print("=",produced_files)
                    found_sess=True
                    found_model_line=0
    '''        
            cnt+=1
    for i,v in enumerate(line_list):
        if i > cnt:
            print("WHAT THE HELL IS THIS ?=",v)
            new_line_list.append(v)
    '''
    return ( new_line_list,found_sess,produced_files)

def find_number_of_fors(ind_,line_list):
    for_counter=0
    for_space=-1
    ret_for_space=-1
    ret_for_ind=None
    first_time=True
    session_for=0
    previous_for=0
    for ind,line in enumerate(line_list):
        if ind<ind_:
            #print("DEBUG in forsssssssssssss=",line)
            num_of_space = len(line) - len(line.lstrip(' '))
            if num_of_space <= for_space and line.isspace()==False and first_time==False:
                print("DEBUG:Find:Exited for = ", line,' __num_of_space= ',num_of_space," ____for_space= ",for_space)
                first_time = True
                for_counter = 0
                ret_for_space = None
                ret_for_ind = None
                for_space=-1
                session_for = 0
            if line.replace(" ","").startswith("for")==True:
                print("DEBUG FIRST TIME IS ", first_time," num_of_space=",num_of_space," for_space=",for_space," matters=",session_for)
                print("File=",line)
                if first_time==True:
                    print("DEBUG:Find:Line with first time for is=",line)
                    first_time=False
                    ret_for_space=num_of_space
                    for_space=ret_for_space
                    ret_for_ind=ind
                    session_for=0
                    previous_for=-1
                    print("DEBUG:Find:Line with num_of_spaces=", ret_for_space)
                if num_of_space>previous_for:
                    previous_for=num_of_space
                    session_for+=1
                elif num_of_space<previous_for:
                    session_for -= 1
                    previous_for=num_of_space
                for_counter+=1
        else:
            break
    print("DEBUG RETURNING=IND",ret_for_ind," For_counter=",for_counter," ret_for_space=",ret_for_space," session that matters ",session_for)
    return (for_counter,ret_for_ind,ret_for_space,session_for)



def handle_feed_dict(line,num_of_space):
    print("\n\n\n\n\n\n\ LINE IS",line)
    feed_dict=line.split("feed_dict=")[1].replace(")","").replace(" ","")
    print("FEED DICT=",feed_dict)
    before_list=[]
    before_list.append((num_of_space) * " " + "f = open('FILE', 'w')\n")
    if "{" in feed_dict:
        before_list.append(num_of_space*" "+"feed_dict="+feed_dict+"\n")
        before_list.append(num_of_space * " " + "for key,value in feed_dict.items():\n")
    else:
        before_list.append(num_of_space * " " + "for key,value in "+feed_dict+".items():\n")
    before_list.append((num_of_space + 1) * " " + "f.write(str(tf.shape(key).shape[0])+'||||')\n")
    before_list.append(num_of_space * " " + "f.close()\n")
    return before_list

def prepare_lists_and_lines(is_co_train,file,session_counter,write_space,file_path):
    print("1ARETH=FILE",file," Session counter=",str(session_counter))
    file=file.replace(".py","")
    if is_co_train == True:
        write_file = "_temporary_" + file + "_" + "_sEssIOn_[" + str(session_counter) + "]_co_train_" + str(
            session_counter) + ".info"
        write_file_line = "_temporary_" + file + "_" + "_sEssIOn_[" + str(session_counter) + "]_co_train_" + str(
            session_counter) + ".lines"
        write_file_batch = "_temporary_" + file + "_" + "_sEssIOn_[" + str(session_counter) + "]_co_train_" + str(
            session_counter) + ".batch"
    else:
        write_file = "_temporary_" + file + "_" + "_sEssIOn_[" + str(session_counter) + "]_" + str(
            session_counter) + ".info"
        write_file_line = "_temporary_" + file + "_" + "_sEssIOn_[" + str(
            session_counter) + "]_" + str(session_counter) + ".lines"
        write_file_batch = "_temporary_" + file + "_" + "_sEssIOn_[" + str(
            session_counter) + "]_"+str(session_counter) + ".batch"
    name = os.path.dirname(file_path).split(github.dirName)[1].replace("\\", "_")
    temp_list = []
    temp_list.append((write_space) * " " + "f = open('" + write_file_line + "', 'a')\n")
    temp_list.append((write_space) * " " + "f.write('" + "----" + "')\n")
    temp_list.append((write_space) * " " + "f.close()\n")
    return (write_file,write_file_line,write_file_batch,temp_list,name)

def append_file_lines(name,line_of_sess_run,new_line_list,temp_list,line_list_for_feed,write_file,write_file_line,write_file_batch,write_ind,before_sess_run,num_of_space,files_written):
    new_line_list = new_line_list[:write_ind] + temp_list + new_line_list[write_ind:]
    line_list_for_feed = [x.replace("FILE", write_file_batch) for x in line_list_for_feed]
    new_line_list = new_line_list[:before_sess_run] + line_list_for_feed + new_line_list[before_sess_run:]
    file = write_file.replace(".info", "")
    new_line_list.append(num_of_space * " " + "abc = ''\n")
    new_line_list.append(num_of_space * " " + "tmp__ = locals().copy()\n")
    new_line_list.append(num_of_space * " " + "for k, v in tmp__.items():\n")
    new_line_list.append((num_of_space + 1) * " " + "abc +='KEY:'+ k + 'VALUE:' + str(v) + '||||'\n")
    new_line_list.append(num_of_space * " " + "f = open('" + write_file + "', 'w')\n")
    new_line_list.append(num_of_space * " " + "f.write(abc)\n")
    new_line_list.append((num_of_space) * " " + "f.close()\n")
    new_line_list.append((num_of_space) * " " + "f = open('" + write_file_line + "', 'a')\n")
    new_line_list.append((num_of_space) * " " + "f.write('" + line_of_sess_run + "||||')\n")
    new_line_list.append((num_of_space) * " " + "f.close()\n")
    return(file,new_line_list)


def create_new_file(line_list,path,file,tf_run_app):
    new_line_list=[]
    current_folder=os.getcwd()
    found_main=0
    return_list=create_code_for_pbtxt_and_tensorboard(path, file)
    if tf_run_app == True:
        found_def_main=False
        spaces=0
        main_func=[]
        last_ind=0
        spaces_to_use=0
        only_once=False
        for ind, line in enumerate(line_list):
            if "def main" in line:
                main_func.append(line)
                found_def_main=True
                spaces=(len(line) - len(line.lstrip(' ')))
            elif found_def_main==True:
                if only_once==False:
                    only_once=True
                    spaces_to_use=len(line) - len(line.lstrip(' '))
                if (len(line) - len(line.lstrip(' ')))>spaces:
                    main_func.append(line)
                else:
                    last_ind=ind
                    for line_ in return_list[1:]:
                        main_func.append(str(spaces_to_use*" "+line_))
                    break
            else:
                new_line_list.append(line)
        for ind, line in enumerate(line_list):
            if ind==last_ind:
                for elem in main_func:
                    new_line_list.append(elem)
                new_line_list.append(line)
            if ind>last_ind:
                new_line_list.append(line)
    else:
        min_space=1000
        for ind,line in enumerate(line_list):
            new_line_list.append(line)
            if "if__name__" in line.replace(" ",""):
                found_main=1
                min_space = (len(line_list[ind+1]) - len(line_list[ind+1].lstrip(' ')))
            if found_main==0 and "tf." in line:
                if  (len(line) - len(line.lstrip(' '))) < min_space:
                    min_space=(len(line) - len(line.lstrip(' ')))
        for elem in return_list:
            new_line_list.append(min_space*" "+elem)
    os.chdir(current_folder)
    return new_line_list

def get_model_name(line):
    model_var = line.split(".")[0]
    return model_var


def model_find(path):
    for line in reversed(list(open(path))):
            if '.run' in line:
                model_variable=get_model_name(line)
                print("Model name is ",model_variable)
                model_find(model_variable)


def create_code_for_pbtxt_and_tensorboard(path,file):
    backwards_dir = ""
    real_path = os.path.dirname(path)
    real_path = os.getcwd() + "\\" + real_path
    os.chdir(real_path)
    while os.path.basename(os.getcwd()) != "thesis":
        os.chdir("..")
        backwards_dir = backwards_dir + "..\\"
    # This happend because the path contains unicode escape characters \a ,\t etc.We needed to make sure the produced path string,
    # would not contain any escape characters.
    num_of_space = 0
    str="\n"
    str_0 = num_of_space * " " + "import tensorflow as tf\n"
    str_1 = num_of_space * " " + "with tf.Session() as sess:\n"
    num_of_space = num_of_space + 1
    str_2 = num_of_space * " " + "graph_def = sess.graph.as_graph_def(add_shapes=True)\n"
    str_3 = num_of_space * " " + "tf.train.write_graph(graph_def,\"" + backwards_dir + github.dirName + "\_tensorflow\pbtxt\"" + ",\"" + file + ".pbtxt\")\n"
    folder = file.replace(".py", "")
    str_4 = num_of_space * " " + "tfFileWriter = tf.summary.FileWriter(\"" + folder + "\")\n"
    str_5 = num_of_space * " " + "tfFileWriter.add_graph(graph_def)\n"
    str_6 = num_of_space * " " + "tfFileWriter.close()\n"
    return_list=[]
    return_list.append(str)
    return_list.append(str_0)
    return_list.append(str_1)
    return_list.append(str_2)
    return_list.append(str_3)
    return_list.append(str_4)
    return_list.append(str_5)
    return_list.append(str_6)
    return (return_list)


def handle_evaluation_score(filePath):
    print("Path is:",filePath)



def end_loop(cnt,line,number_of_fors,found_total_session):
    if found_total_session==False:
        if number_of_fors==cnt:
            return True
    if ".total_session" in line and "= open(" in line:
        print("DONE DUE TO TOTAL SESSION=",line)
        return True
    return False