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
    with open(new_file, 'a',errors="replace") as f:
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
        print("ERROR:File:",file_to_execute ,"\nTraceback:",traceback)
        if "batch" in traceback:
            return "batch"
        else:
            return "error"

def parse_file(path,skip,tf_run_app):
    path_to_folder=os.path.dirname(path)
    file=ntpath.basename(path)
    new_file=path_to_folder+"/summary_"+file
    new_file=new_file[:-1]
    print("LOGGING:Created new temporary file to run with needed changes in path ",new_file)
    line_list=[]
    for line in open(path,errors="replace"):
        line_list.append(line)
    #In case there is a meaningful last line,to reduce additional checks for last line
    #line_list.append(" ")
    (pbtxt_file,batch_epoch,model_var,new_line_list)=create_new_file(line_list,path,file,skip,tf_run_app)
    new_line_list=find_epoch_size(new_line_list)
    result=execute_file(new_file,new_line_list,path_to_folder)
    if result=="error":
        print("ERROR:File contains inner error,cannot execute it.")
        return ("error", None, None, None)
    else:
        print("LOGGING:Successfully executed.")
        (batch_size,epoch)=get_batch_epoch(batch_epoch)
        if batch_size==-2 and epoch==-2:
            print("ERROR:Could not obtain epoch/batch information")
            return ("success",pbtxt_file,batch_size,epoch)
        else:
            return ("success",pbtxt_file,batch_size,epoch)

def get_batch_epoch(file):
    batch_size=-1
    epoch=-1
    try:
        for line in open(file):
            if "BATCH SIZE" in line:
                batch_size=int(line.split(":")[1])
            elif "EPOCH COUNTER" in line:
                epoch=int(line.split(":")[1])
        if epoch==-1:
            return (-3,-3)
        return(batch_size,epoch)
    except FileNotFoundError:
        return (-2,-2)

def find_epoch_size(line_list):
    new_line_list = []
    cnt=0
    found_model_line=0
    num_of_space=0
    for ind,line in enumerate(line_list):
        new_line_list.append(line)
        if ".run" in line:
            num_of_space = len(line) - len(line.lstrip(' '))
        if "feed_dict" in line:
            found_model_line = 1
        if line.replace("\n", "").endswith(')') == True  and found_model_line == 1:
            if ind+1 >=len(line_list) or line_list[ind + 1][0].replace(" ","") != ".":
                new_line_list.append(num_of_space*" "+"global epoch_counter\n")
                new_line_list.append(num_of_space*" "+"epoch_counter+=1\n")
                new_line_list.append(num_of_space*" "+"print(\"EPOCH COUNTER:\",epoch_counter,file=batch_epoch_file_to_print)\n")
                break
        cnt+=1
    for i,v in enumerate(line_list):
        if i > cnt:
            new_line_list.append(v)
    return ( new_line_list)

def create_new_file(line_list,path,file,skip,tf_run_app):
    new_line_list=[]
    model_variable=""
    batch_epoch=file+"_batch_epoch.txt"
    current_folder=os.getcwd()
    first_time=0
    found_main=0
    (return_list,pbtxt_file)=create_code_for_pbtxt_and_tensorboard(path, file, skip)
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
    return (pbtxt_file,batch_epoch,model_variable, new_line_list)

def get_model_name(line):
    model_var = line.split(".")[0]
    return model_var


def model_find(path):
    for line in reversed(list(open(path))):
            if '.run' in line:
                model_variable=get_model_name(line)
                print("Model name is ",model_variable)
                model_find(model_variable)


def create_code_for_pbtxt_and_tensorboard(path,file,skip):
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
    if "batch" not in skip:
        str_7 = num_of_space * " " + "if 'batch_size' in locals():\n"
        str_8 = (num_of_space + 1) * " " + "print(\"BATCH SIZE:\",batch_size,file=batch_epoch_file_to_print)\n"
    return_list=[]
    return_list.append(str)
    return_list.append(str_0)
    return_list.append(str_1)
    return_list.append(str_2)
    return_list.append(str_3)
    return_list.append(str_4)
    return_list.append(str_5)
    return_list.append(str_6)
    return_list.append(str_7)
    return_list.append(str_8)
    pbtxt_file=backwards_dir+github.dirName+"\_tensorflow\pbtxt\\"+file+".pbtxt"
    return (return_list,pbtxt_file)


def handle_evaluation_score(filePath):
    print("Path is:",filePath)
