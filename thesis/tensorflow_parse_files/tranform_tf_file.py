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
        return "error"

#Start the handling of the files that are not main files.
def handle_imported_files(proj,prod_files):
    print("LOGGING:Started searching for project included files")
    for path in proj:
        line_list=[]
        for line in open(path, errors="replace"):
            line_list.append(line)
        dir_ = os.getcwd()
        os.chdir(os.path.dirname(str(path)))
        #Handle each function file and find if there is a sess run and place lines accordingly.
        (new_line_list,found,produced_files_one_file)=find_epoch_size(line_list,str(path))
        #Keep produced file names.
        for produced_file in produced_files_one_file:
            prod_files.append(produced_file)
        import ntpath
        real_path=ntpath.basename(str(path))
        if found==True:
            f=open(real_path,'w')
            print("LOGGING:Finished tranforming file,writing to file ",real_path," get=",os.getcwd())
            for elem in new_line_list:
                f.write(elem)
            f.close()
        os.chdir(str(dir_))
    return prod_files

#Parse file will parse the python file and it will replace the existing one with the new one that include changes
#to force the print of information to obtain the graph and the training info
def parse_file(path,tf_run_app,proj):
    path_to_folder=os.path.dirname(path)
    file=ntpath.basename(path)
    new_file=path_to_folder+"/"+file
    #If the file was already executed,we just retrieve the file list of the produced files.
    (no_execution,line_list,produced_files)=check_if_already_executed(path,proj)
    if no_execution==True:
        return ("no execution",list(set(produced_files)))
    #Here we perform the creation of the new file.
    new_line_list=create_new_file(line_list,path,file,tf_run_app)
    dir_ = os.getcwd()
    os.chdir(os.path.dirname(str(path)))
    #Start the transformation of the main file,in case a sess run is presented.
    (new_line_list,_,produced_files)=find_epoch_size(new_line_list,path)
    os.chdir(str(dir_))
    #Do the same for the files imported into
    produced_files=handle_imported_files(proj,produced_files)
    #Execute file.
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
            prod_files.append(line.split('\'')[1].split('.')[0])
            no_execution = True
        line_list.append(line)
    for filepath in proj:
        for ind, line in enumerate(open(filepath, errors="replace")):
            if "_sEssIOn_" in line:
                prod_files.append(line.split('\'')[1].split('.')[0])
                no_execution = True

    return (no_execution,line_list,prod_files)



def find_epoch_size(line_list,file_path):
    #Multiple variables needed for the file handling.
    new_line_list = []
    found_model_line=0
    num_of_space=0
    found_sess=False
    line_of_sess_run=""
    file=file_path.split("\\")[2:]
    file="_".join(file)
    found_run=False
    produced_files=[]
    multiple_line_comm=False
    session_counter=0
    added_lines=0
    before_sess_run=0
    files_added=0
    found_total_session=False
    first_for_space=100000000
    close_=""
    last_line=False
    last_line_run=False
    multiple_line_sentence_counter=0
    multiple_line_sentence_space=0
    files_replace=[]
    done_with_continuous=True
    #######################################

    for ind_,line in enumerate(line_list):
        new_line_list.append(line)
        #Number of spaces before the first letter(indentation)
        line_space=len(line) - len(line.lstrip(' '))
        #In case of empty line,continue with the next line.
        if line.isspace() == True or line.replace(" ","").startswith("#")==True:
            continue
            #if line has less right or left parenthesis and we are not alredy in multiple line,make variable true
            #and save line counter of the multiple line sentence
        if (line.replace(" ","").endswith("\\") or line.count("(")!=line.count(")")) and multiple_line_comm==False:
            multiple_line_comm=True
            multiple_line_sentence_counter=len(new_line_list)-1
            multiple_line_sentence_space=line_space
            done_with_continuous=False
        #Check if we are in multiple line sentence  and if this mutiple line is sentence based on the indentation
        elif multiple_line_comm==True and multiple_line_sentence_space>=line_space:
            new_line_list[-1]=new_line_list[-1]+"\n"
            done_with_continuous=True
            multiple_line_comm=False
            if (line.replace(" ", "").endswith("\\") or line.count("(") != line.count(
                ")")) and multiple_line_comm == False:
                multiple_line_comm = True
                multiple_line_sentence_counter = len(new_line_list) - 1
                multiple_line_sentence_space = line_space
                done_with_continuous = False
        if ind_ == len(line_list) - 1:
            last_line = True
            #Found total session indicated that we have found a largest session containing the steps,the indentation changed,
            #thus the file opened for session(to write session epochs) needs to be closed
        if found_total_session==True:
            if first_for_space>=line_space and (last_line==True or for_counter!=0) or first_for_space>line_space:
                found_total_session = False
                added_lines+=1
                if (found_run==True or ".run(" in line) and last_line==True:
                    last_line_run=True
                else:
                    new_line_list = new_line_list[:-1] + [close_] + new_line_list[-1:]
                    close_=""
        #If we found a .run line and it is not part of a multiline command,we have to set the value
        #to false.
        if found_run==True and multiple_line_comm==False:
            found_run=False
        #Case of .run:
        if ".run" in line:
            #Store some information for the sess.run line.It requires different handling for commands extending beyond
            #one line.
            if multiple_line_comm==False:
                #If false,keep current line information.
                #Indentation
                num_of_space = len(line) - len(line.lstrip(' '))
                #Line text
                line_of_sess_run=line
                found_run=True
                #Added lines counter
                added_lines = len(new_line_list) - ind_ - 1
                #Keep current index.
                before_sess_run=ind_+added_lines
            else:
                num_of_space = multiple_line_sentence_space
                line_of_sess_run = line
                found_run = True
                before_sess_run = multiple_line_sentence_counter
        if found_run==True:
            #If continuous line command reached line where brackets are closed.
            if done_with_continuous==False and line not in line_of_sess_run:
                line_of_sess_run+=line
            if "feed_dict" in line:
                found_model_line = 1
        #If we found sess run with a feed dict
        if found_model_line == 1:
            if (line.replace("\n", "").endswith(')')== True or "}" in line):
                line_of_sess_run=line_of_sess_run.replace("\n", "").replace(" ", "")
                #Line that contains feed dict,we handle it and retrieve variables such as optimizer,input,batch variable
                line_list_for_feed = handle_feed_dict(line_of_sess_run, num_of_space)
                #If this is the last line and it does not end with a dot,meaning next line is part of this line.
                if ind_+1 >=len(line_list) or line_list[ind_ + 1][0].replace(" ","") != ".":
                    added_lines = len(new_line_list) - ind_-1
                    temp_ind = ind_ + added_lines
                    prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                    is_co_train=False
                    session_counter+=1
                    for_counter=0
                    write_ind=0
                    write_space=0
                    first_time=False
                    #First time we encounter a run,start of a new session.
                    if found_total_session==False:
                        (number_of_fors,session_for_ind,first_for_space,session_fors)=find_number_of_fors(temp_ind,new_line_list)
                    no_rep=False
                    temp_ind=before_sess_run-1
                    while end_loop(for_counter,new_line_list[temp_ind],number_of_fors,found_total_session) ==False and no_rep==False:
                        #If feed dict is encountered in previous lines before current sess run and the indentation
                        # of this sess run is different,we need to make the variable for co
                        #train as true.That indicates that there is another neural network being trained in the same loop
                        # as part of the annett-o entity for training loop.We will need to name the training steps that
                        #are primary/simple with a different name in order to distinguish them.
                        if "feed_dict=" in new_line_list[temp_ind] and num_of_space!=prev_line_space \
                                and ")" in new_line_list[temp_ind]:
                            is_co_train=True
                        #Each file name contains a number,the identification that will be used in order to separate
                        #which training step belongs to which session.Each name will have a prefix that will be the path
                        #to the file,it will be followed with a number to show which training steps are co-trained or belong
                        #to the same session.
                        if "_sEssIOn_" in new_line_list[temp_ind]:
                            file_replace=new_line_list[temp_ind].split('\'')[1].split('.')[0]
                            reg = r'\[[\s\S]*\]'
                            new_line_list[temp_ind] = re.sub(reg, '[' + str(session_counter) + ']',
                                                             new_line_list[temp_ind])
                            new_file = new_line_list[temp_ind].split('\'')[1].split('.')[0]
                            if file_replace not in files_replace:
                                produced_files.remove(file_replace)
                                files_replace.append(file_replace)
                                produced_files.append(new_file)
                        #No for before sess.run case
                        if session_fors == 0:
                            if "total_session_abc=open" in new_line_list[temp_ind - 1].replace(" ","") or found_total_session==False:
                                first_time = True
                                no_rep = True
                                write_space = first_for_space
                                write_ind = session_for_ind+1
                        #For encountered,need increase counter of fors.
                        elif new_line_list[temp_ind].replace(" ", "").startswith("for"):
                            for_counter += 1
                            for_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                            #Check whether the indentation of for is smaller than the indentation for sess run
                            #if yes and first time for for is true we save the index of the line.The index will be either
                            #the index(if the session was found) or plus 1 if the session is currently been processed(one extra
                            #line added for opening the file for writing for a session)
                            if num_of_space > for_space and first_time==False:
                                first_time=True
                                if found_total_session==False:
                                    write_ind = temp_ind+1
                                else:
                                    write_ind = temp_ind
                                write_space = len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                        #Main loop iterator.We iterate backwards.
                        temp_ind-=1
                        prev_line_space=len(new_line_list[temp_ind]) - len(new_line_list[temp_ind].lstrip(' '))
                    #We encountered new session
                    if found_total_session==False:
                        found_total_session=True
                        total_session = "_temporary_" + file.replace(".py","") + "_" + "_sEssIOn_[" + str(
                            session_counter) + "]_" + str(
                            session_counter) + ".total_session"
                        if session_fors == 1:
                            temp_sess = session_for_ind
                        else:
                            temp_sess=session_for_ind+1
                        if "total_session_abc.close()" in new_line_list[session_for_ind]:
                            session_for_ind+=1
                        #Skip the empty lines.
                        while new_line_list[temp_sess].isspace()==True:
                            temp_sess+=1
                        temp_space= len(new_line_list[temp_sess]) - len(new_line_list[temp_sess].lstrip(' '))
                        produced_files.append(total_session)
                        #New file to write session info
                        open_=(first_for_space)* " " + "total_session_abc = open('" + total_session + "', 'w')\n"
                        #For every session required to train neural network(s),we write ----| in the file.
                        #Later it will be used to perform a split to the string,in order to count the amount of sessions
                        #executed.
                        write_=(temp_space)* " " + "total_session_abc.write('----|')\n"
                        #Close session file.
                        close_=(first_for_space)* " " + "total_session_abc.close()\n"
                        #Place to append the opening of the file .It will be placed before the outer for
                        new_line_list = new_line_list[:session_for_ind] + [open_] + new_line_list[session_for_ind:]
                        #Place the writing of the file before inner for.
                        new_line_list = new_line_list[:temp_sess+1] + [write_] + new_line_list[ temp_sess+1:]
                        if session_fors%2!=1:
                            write_ind += 1
                        #5 added lines ,need to be included in the new index pointing at line with the sess.run
                        before_sess_run+=5
                    else:
                        reg = r'\[[\s\S]*\]'
                        new_line_list[temp_ind] = re.sub(reg, '[' + str(session_counter) + ']',
                                                         new_line_list[temp_ind])
                        before_sess_run += 3
                    files_added+=1
                    #Prepare the other file lines.
                    (write_file, write_file_line, write_file_batch,temp_list, name)=prepare_lists_and_lines(is_co_train,file,session_counter,write_space,file_path)
                    (prod_file,new_line_list)=append_file_lines(name,line_of_sess_run, \
                                      new_line_list,temp_list,line_list_for_feed, \
                                      write_file,write_file_line,write_file_batch,write_ind,
                                      before_sess_run,num_of_space,files_added)
                    produced_files.append(prod_file)
                    found_sess=True
                    found_model_line=0
    if last_line_run==True or close_!="":
        new_line_list.append("\n" + close_)
    return ( new_line_list,found_sess,produced_files)

def find_number_of_fors(ind_,line_list):
    for_counter=0
    for_space=-1
    ret_for_space=-1
    ret_for_ind=None
    first_time=True
    session_for=0
    fors_list_len=[]
    prev_num=-1
    prev_ind=-1
    for ind,line in enumerate(line_list):
        #Iterate until the line the sess run occured.
        if ind<=ind_:
            num_of_space = len(line) - len(line.lstrip(' '))
            tmp_line=line.replace(" ","")
            #Any change of the indentation needs to be monitored.
            if tmp_line.startswith("if") or tmp_line.startswith("with"):
                prev_num=num_of_space
                prev_ind=ind-1
            #Fors_list_len has the number of spaces per each for.
            if fors_list_len!=[] and num_of_space<=fors_list_len[-1] and line.isspace()==False:
                del fors_list_len[-1]
                session_for -= 1
            #We need to store basic information regarding the training of the network.
            #We must find the number of spaces per space(we will store the previous number of spaces of the last for),
            #the number of fors before sess.run and other information.
            if num_of_space <= for_space and line.isspace()==False and first_time==False:
                first_time = True
                for_counter = 0
                ret_for_space = num_of_space
                ret_for_ind = ind
                for_space=-1
                session_for = 0
            if line.replace(" ","").startswith("for")==True:
                if first_time==True:
                    first_time=False
                    ret_for_space=num_of_space
                    for_space=ret_for_space
                    ret_for_ind=ind
                    fors_list_len=[]
                    session_for=0
                if fors_list_len==[] or num_of_space>fors_list_len[-1]:
                    fors_list_len.append(num_of_space)
                    session_for+=1
                elif num_of_space<fors_list_len[-1]:
                    del fors_list_len[-1]
                    session_for -= 1
                for_counter+=1
        else:
            break
    #No for before sess.run
    if ret_for_ind==None:
        #print("LOGGING:Returning index for first for is none,no for presented.Line is ",line_list[prev_ind])
        ret_for_space=prev_num
        ret_for_ind=prev_ind
    #print("DEBUG RETURNING=IND",ret_for_ind," For_counter=",for_counter," ret_for_space=",ret_for_space," session that matters ",session_for)
    return (for_counter,ret_for_ind,ret_for_space,session_for)



def handle_feed_dict(line,num_of_space):
    #If feed dict is not present,it means there is a variable as a second argument.
    if "feed_dict=" not in line.replace(" ",""):
        feed_dict=line.replace(" ", "").split(",")[-1].replace(")","")
    #Else split the line based on feed dict.
    else:
        feed_dict = line.split("feed_dict=")[1]
        #If } is present,then argument of feed dict is of type
        #feed_dict={a,x,v...,b}
        if "}" in feed_dict:
            feed_dict=feed_dict.split("}")[0].replace(" ","")+"}"
        #Else the variable name is feed_dict
        else:
            feed_dict=feed_dict.replace(")","")
    before_list=[]
    #We need to add some lines in order to be able tp retrieve the variables into feed dict.
    #Thus we need just before the sess run to add the following lines.
    #a)Open file
    #b)Write dictionary elements into a file
    #This is how we store the dictionary elements.
    before_list.append((num_of_space) * " " + "mYFiLe = open('FILE', 'w')\n")
    if "{" in feed_dict:
        before_list.append(num_of_space*" "+"FEED_DICT="+feed_dict+"\n")
        before_list.append(num_of_space * " " + "for key,value in FEED_DICT.items():\n")
    else:
        before_list.append(num_of_space * " " + "for key,value in "+feed_dict+".items():\n")
    before_list.append((num_of_space + 1) * " " + "if 'numpy.ndarray' in str(type(value)):\n")
    before_list.append((num_of_space + 2) * " " + "mYFiLe.write(str(len(value))+'||||')\n")
    before_list.append((num_of_space + 1) * " " + "else:\n")
    before_list.append((num_of_space + 2) * " " + "mYFiLe.write(str(0)+'||||')\n")
    before_list.append(num_of_space * " " + "mYFiLe.close()\n")
    return before_list

def prepare_lists_and_lines(is_co_train,file,session_counter,write_space,file_path):
    file=file.replace(".py","")
    #If co train variable is true we need to change the name of the files to be written.Co train means there is a training loop.
    #   -.info file: Will contain a dictionary with all the variables up until this point of the program.
    #   -.lines file: Number of session represented by ----,the line that has sess.run
    #   -.batch file: Feed dict inputs value. Up to this point the placeholder would contain the batch,thus the dimension regarding
    #               batch size would be accesible.
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
    temp_list.append((write_space) * " " + "mYFiLe = open('" + write_file_line + "', 'a')\n")
    temp_list.append((write_space) * " " + "mYFiLe.write('" + "----" + "')\n")
    temp_list.append((write_space) * " " + "mYFiLe.close()\n")
    return (write_file,write_file_line,write_file_batch,temp_list,name)

def append_file_lines(name,line_of_sess_run,new_line_list,temp_list,line_list_for_feed,write_file,write_file_line,write_file_batch,write_ind,before_sess_run,num_of_space,files_written):
    #Append the created lists.
    new_line_list = new_line_list[:write_ind] + temp_list + new_line_list[write_ind:]
    line_list_for_feed = [x.replace("FILE", write_file_batch) for x in line_list_for_feed]
    new_line_list = new_line_list[:before_sess_run] + line_list_for_feed + new_line_list[before_sess_run:]
    file = write_file.replace(".info", "")
    new_line_list.append(num_of_space * " " + "abc = ''\n")
    new_line_list.append(num_of_space * " " + "tmp__ = locals().copy()\n")
    new_line_list.append(num_of_space * " " + "for k, v in tmp__.items():\n")
    new_line_list.append((num_of_space + 1) * " " + "abc +='KEY:'+ k + 'VALUE:' + str(v) + '||||'\n")
    new_line_list.append(num_of_space * " " + "mYFiLe = open('" + write_file + "', 'w')\n")
    new_line_list.append(num_of_space * " " + "mYFiLe.write(abc)\n")
    new_line_list.append((num_of_space) * " " + "mYFiLe.close()\n")
    new_line_list.append((num_of_space) * " " + "mYFiLe = open('" + write_file_line + "', 'a')\n")
    new_line_list.append((num_of_space) * " " + "mYFiLe.write('" + line_of_sess_run.replace("'","\\'") + "||||')\n")
    new_line_list.append((num_of_space) * " " + "mYFiLe.close()\n")
    return(file,new_line_list)

#Lines for creation of pbtxt file ,plus file as a list.
def create_new_file(line_list,path,file,tf_run_app):
    new_line_list=[]
    current_folder=os.getcwd()
    found_main=0
    #Create the lines for the pbtxt file.
    return_list=create_code_for_pbtxt_and_tensorboard(path, file)
    if tf_run_app == True:
        found_def_main=False
        spaces=0
        main_func=[]
        last_ind=0
        spaces_to_use=0
        only_once=False
        found_import_tf=False
        #Based on case encountered,there is a different indentation/line number for the pbtxt file creation lines
        #to be places.
        for ind, line in enumerate(line_list):
            if "import" in line and "tensorflow" in line:
                found_import_tf=True
            if "def main" in line:
                main_func.append(line)
                found_def_main=True
                spaces=(len(line) - len(line.lstrip(' ')))
            elif found_def_main==True:
                if only_once==False:
                    only_once=True
                    spaces_to_use=len(line) - len(line.lstrip(' '))
                if (len(line) - len(line.lstrip(' ')))>spaces or line.isspace()==True:
                    main_func.append(line)
                else:
                    last_ind=ind
                    if found_import_tf==True:
                        for line_ in return_list[2:]:
                            main_func.append(str(spaces_to_use*" "+line_))
                    else:
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

#Python code for pbtxt into specific folder _tensorflow\pbtxt with main file name+pbtxt.
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

#Function to check if the we must break tha main loop of the loop.
def end_loop(cnt,line,number_of_fors,found_total_session):
    #If this is the first for found
    if found_total_session==False:
        #If the numbers of for before sess matched the ones we encounter while iterating over the line list
        if number_of_fors==cnt and number_of_fors!=0:
            return True
    #If we hit the start of the writing for current session
    if ".total_session" in line and "= open(" in line:
        return True
    #Previous session,closing of file used for previous session
    elif "total_session_abc.close()" in line:
        return True
    return False