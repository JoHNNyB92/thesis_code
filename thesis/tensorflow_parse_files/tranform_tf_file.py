import re
from shutil import copy
import shlex, subprocess
import os
import ntpath
import sys
import github.github as github

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
        traceback = str(e)
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
        (new_line_list,found,produced_files)=find_epoch_size(line_list,str(path))
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
    #new_file=new_file[:-1]
    print("LOGGING:Created new temporary file to run with needed changes in path ",new_file)
    line_list=[]
    no_execution=False
    produced_files=[]
    for ind,line in enumerate(open(path,errors="replace")):
        if "_sEssIOn_" in line:
            print("LINE=",line)
            print("LINE=",line.split('\'')[1])
            produced_files.append(line.split('\'')[1].split('.')[0])
            no_execution=True
        line_list.append(line)
    if no_execution==True:
        return ("no execution",list(set(produced_files)))
    #In case there is a meaningful last line,to reduce additional checks for last line
    #line_list.append(" ")
    (pbtxt_file,model_var,new_line_list)=create_new_file(line_list,path,file,tf_run_app)
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

def find_epoch_size(line_list,file_path):
    new_line_list = []
    print("Start iterating")
    for ind,elem in enumerate(line_list):
        print("\n\n\n\n\nelem=",elem)
        new_line_list.append(elem)
        if elem.startswith("from") or elem.startswith("import"):
            if "__future__" not in line_list:
                break
    print("End iterating")
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
    for ind_,line in enumerate(line_list):
        if ind_>ind:
            new_line_list.append(line)
            #num_of_space = len(line) - len(line.lstrip(' '))
            if ".run" in line:
                num_of_space = len(line) - len(line.lstrip(' '))
                print("Found run=",num_of_space)
                line_of_sess_run=line
                found_run=True
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
                        temp_ind=ind_+added_lines
                        temp_ind_copy=temp_ind
                        sess_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                        #temp_ind-=1
                        prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                        is_co_train=False
                        #print("LINE IS=",line_of_sess_run)
                        #print("LINE2 IS=",new_line_list[temp_ind])
                        #print("ALOHA=",sess_space,"-",prev_line_space,"-",new_line_list[temp_ind])
                        session_counter+=1
                        for_counter=0
                        write_ind=0
                        write_space=0
                        previous_for=10000
                        files_replace=[]
                        while end_loop(for_counter,new_line_list[temp_ind])==False:
                            if "sess.run" in new_line_list[temp_ind] and num_of_space<prev_line_space:
                                is_co_train=True
                            if "_sEssIOn_" in new_line_list[temp_ind]:
                                #Line format:
                                #f = open('_temporary_path_[x].batch', 'w')
                                files_replace.append(new_line_list[temp_ind].split('\'')[1].split('.')[0])
                                print("FOUND SESSION WITH REGEXP=",new_line_list[temp_ind])
                                reg = r'\[[\s\S]*\]'
                                new_line_list[temp_ind]=re.sub(reg, '['+str(session_counter)+']', new_line_list[temp_ind])
                                #print("FOUND SESSION WITH=", new_line_list[temp_ind])

                            if num_of_space > prev_line_space:
                                print("1ELENHLINE=", new_line_list[temp_ind])
                                if prev_line_space<previous_for:
                                    print("2ELENHLINE=", new_line_list[temp_ind])
                                    if new_line_list[temp_ind].replace(" ", "").startswith("for"):
                                        print("3ELENHLINE=",new_line_list[temp_ind])
                                        for_counter += 1
                                        write_ind=temp_ind
                                        previous_for=prev_line_space
                                        write_space=prev_line_space
                            temp_ind-=1
                            prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                        temp_ind_copy=temp_ind
                        while temp_ind_copy>0:
                            for file_to_be_replaced in set(files_replace):
                                if file_to_be_replaced in new_line_list[temp_ind_copy]:
                                    print("re file=",file_to_be_replaced," ",new_line_list[temp_ind_copy])
                                    print("FOUND OUTER SESSION WITH REGEXP=", new_line_list[temp_ind_copy])
                                    produced_files.remove(file_to_be_replaced)
                                    reg = r'\[[\s\S]*\]'
                                    new_line_list[temp_ind_copy] = re.sub(reg, '[' + str(session_counter) + ']',new_line_list[temp_ind_copy])
                                    produced_files.append(new_line_list[temp_ind_copy].split('\'')[1].split('.')[0])
                            temp_ind_copy-=1
                        files_added+=1
                        (write_file, write_file_line, write_file_batch,temp_list, name)=prepare_lists_and_lines(is_co_train,file,session_counter,write_space,file_path)
                        (prod_file,new_line_list)=append_file_lines(name,line_of_sess_run, \
                                          new_line_list,temp_list,line_list_for_feed, \
                                          write_file,write_file_line,write_file_batch,write_ind,
                                          before_sess_run,num_of_space,files_added)
                        added_lines=added_lines+len(line_list_for_feed)+10
                        produced_files.append(prod_file)
                        print("2ARETH=",produced_files)
                        found_sess=True
                        found_model_line=0
            cnt+=1
    for i,v in enumerate(line_list):
        if i > cnt:
            new_line_list.append(v)

    return ( new_line_list,found_sess,produced_files)


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
        before_list.append((num_of_space) * " " + "f = open('FILE', 'w')\n")
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
    new_line_list = new_line_list[:before_sess_run + files_written*3] + line_list_for_feed + new_line_list[before_sess_run + files_written*3:]
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
    model_variable=""
    current_folder=os.getcwd()
    first_time=0
    found_main=0
    (return_list,pbtxt_file)=create_code_for_pbtxt_and_tensorboard(path, file)
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
            if first_time == 0 and "import tensorflow" in line or "from tensorflow." in line or "from tensorflow import" in line:
                first_time=1
                spaces = len(line) - len(line.lstrip(' '))
                new_line_list.append(spaces * " " + "global epoch_counter\n")
                new_line_list.append(spaces * " " + "epoch_counter=0\n")
                str_1 = spaces * " " + "batch_epoch_file_to_print = open(\"" + file + "_batch_epoch.txt\", \'w\')\n"
                new_line_list.append(str_1)
        for elem in return_list:
            new_line_list.append(min_space*" "+elem)
    os.chdir(current_folder)
    return (pbtxt_file,model_variable, new_line_list)

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
    pbtxt_file=backwards_dir+github.dirName+"\_tensorflow\pbtxt\\"+file+".pbtxt"
    return (return_list,pbtxt_file)


def handle_evaluation_score(filePath):
    print("Path is:",filePath)



def end_loop(cnt,line):
    if cnt==2:
        return True
    elif len(line) - len(line.lstrip(' '))==0 and line.startswith("for")==False:
        return True
    else:
        tmp_line=line
        tmp_line=tmp_line.replace(" ","").lower()
        if tmp_line.startswith("def") or tmp_line.startswith("withtf.session"):
            return True
    return False