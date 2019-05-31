import re
from shutil import copy
import shlex, subprocess
import os
import ntpath
import sys


def parse_file(path):
    path_to_folder=os.path.dirname(path)
    file=ntpath.basename(path)
    new_file=path_to_folder+"/summary_"+file
    new_file=new_file[:-1]
    print(new_file)
    line_list=[]
    for line in open(path):
        line_list.append(line)
    #in case there is a meaningful last line,to reduce additional checks for last line
    #line_list.append(" ")
    (pbtxt_file,batch_epoch,model_var,new_line_list)=create_new_file(line_list,path,file)
    new_line_list=find_epoch_size(new_line_list)
    try:
        os.remove(new_file)
    except OSError:
        pass
    with open(new_file, 'a') as f:
        for line in new_line_list:
            f.write(line)
    file_to_execute = ntpath.basename(new_file)
    os.chdir(path_to_folder)
    theproc = subprocess.Popen([sys.executable, file_to_execute])
    theproc.communicate()
    (batch_size,epoch)=get_batch_epoch(batch_epoch)
    return (pbtxt_file,batch_size,epoch,file_to_execute)

def get_batch_epoch(file):
    batch_size=0
    epoch=0
    for line in open(file):
        if "BATCH SIZE" in line:
            batch_size=int(line.split(":")[1])
        elif "EPOCH COUNTER" in line:
            epoch=int(line.split(":")[1])
    return(batch_size,epoch)

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

def create_new_file(line_list,path,file):
    new_line_list=[]
    model_variable=""
    batch_epoch=file+"_batch_epoch.txt"
    current_folder=os.getcwd()
    first_time=0
    min_space=10000
    found_main=0
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

    backwards_dir = "..\\..\\"
    real_path = os.path.dirname(path)
    real_path = os.getcwd() + "\\" + real_path
    os.chdir(real_path)
    #Original folder of project-diplo
    while os.path.basename(os.getcwd())!="diplo":
        os.chdir("..")
        backwards_dir=backwards_dir+"..\\"
    #This happend because the path contains unicode escape characters \a ,\t etc.We needed to make sure the produced path string,
    #would not contain any escape characters.

    num_of_space = min_space
    #str_1=num_of_space*" "+"f = open(\""+file+"_batch_epoch.txt\", \'w\')\n"
    str_1="\n"+num_of_space*" "+"with tf.Session() as sess:\n"
    num_of_space=num_of_space+1
    str_2=num_of_space*" "+"graph_def = sess.graph.as_graph_def(add_shapes=True)\n"
    str_3=num_of_space*" "+"tf.train.write_graph(graph_def,\" "+backwards_dir+"_git_repositories\_topologies\_tensorflow\pbtxt\""+",\""+file+".pbtxt\")\n"
    folder=file.replace(".py","")
    str_4=num_of_space*" "+"tfFileWriter = tf.summary.FileWriter(\""+folder+"\")\n"
    str_5=num_of_space*" "+"tfFileWriter.add_graph(graph_def)\n"
    str_6=num_of_space*" "+"tfFileWriter.close()\n"
    #str_7=num_of_space*" "+"epoch_counter=0\n"
    str_8=num_of_space*" "+"print(\"BATCH SIZE:\",batch_size,file=batch_epoch_file_to_print)\n"

    new_line_list.append(str_1)
    new_line_list.append(str_2)
    new_line_list.append(str_3)
    new_line_list.append(str_4)
    new_line_list.append(str_5)
    new_line_list.append(str_6)
    #new_line_list.append(str_7)
    new_line_list.append(str_8)

    os.chdir(current_folder)
    pbtxt_file=backwards_dir+"_git_repositories\_topologies\_tensorflow\pbtxt\\"+file+".pbtxt"
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
