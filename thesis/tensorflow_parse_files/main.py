from github import github
import os
import sys
import tranform_tf_file
import tensorflow_parser
import handle_lines_and_info
import csv
from modulefinder import ModuleFinder

def handle_main_file(file,subdir,file_counter):
    found_network=False
    total_path = os.path.join(subdir, file)
    result=""
    handler_entities=""
    if str(file).endswith('.py'):
        print("LOGGING:Examining main file ",str(file))
        with open(file, encoding="utf8", errors='replace') as myfile:
            has_sess = False
            has_run=False
            is_main=False
            has_tf_app_run=False
            has_def_main=False
            has_interactive=False
            if "tf.InteractiveSession" in myfile.read():
                has_interactive=True
            myfile.seek(0)
            if '.run(' in myfile.read():
                has_run=True
            myfile.seek(0)
            if  'tf.Session' in myfile.read():
                has_sess = True
            myfile.seek(0)
            if '__main__' in myfile.read():
                is_main = True
            myfile.seek(0)
            if 'tf.app.run' in myfile.read():
                has_tf_app_run = True
            myfile.seek(0)
            if "def main(" in myfile.read():
                has_def_main=True
            produced_files=[]
            if (has_sess==True and has_run==True) or is_main==True or (has_def_main==True and has_tf_app_run==True) or has_interactive==True:
                path = os.getcwd()
                tf_run_app=False
                if (has_def_main==True and has_tf_app_run==True):
                    tf_run_app=True
                (result,produced_files)= tranform_tf_file.parse_file(file,tf_run_app,project_structure)
                os.chdir(path)
                if "error" in result:
                    print("ERROR:Error occured when executing the program ", file)
                    import sys
                    sys.exit()
                else:
                    print("LOGGING:Finished executing file :", os.path.basename(total_path))
                    os.chdir(path)
                    print("----------------------------------------------------------------------------")
                    # Change directory in order to be appropriate for the folder that the pbtxt parser is located.
                    pbtxt_file=github.folder + github.dirName +"\\_tensorflow\pbtxt\\"+os.path.basename(total_path)+".pbtxt"
                    # Windows OS
                    log_file = pbtxt_file.split("\\")[-1].replace(".py.pbtxt", "")
                    # Unix OS
                    log_file = log_file.split("/")[-1].replace(".py.pbtxt", "")
                    log_file = "../log/" + log_file
                    if os.path.exists(pbtxt_file)==False:
                        print("ERROR:There was an error with the creation of pbtxt file  ",pbtxt_file)
                    else:
                        print("LOGGING:Begin parsing pbtxt file ", pbtxt_file, " with anneto logging in ", log_file)
                        (result,handler_entities)= tensorflow_parser.begin_parsing(os.path.basename(subdir), pbtxt_file,log_file,file_counter)
                        if result == "success":
                            found_network = True
                        print(
                            " ----------------------------------------------------------------------------------------------------------------------")
                        print("LOGGING:|Finished parsing of file ", os.path.basename(total_path), " Result:", result, "|")

                        print(
                            " ---------------------------------------------------------------------------------------------------------------------")
    return(result,found_network,handler_entities,produced_files)

def handle_dotted_imports(import_):
    path=[]
    for elem in import_.split("."):
        path.append(elem)
    return path

def find_imports(toCheck):
    """
    Given a filename, returns a list of modules imported by the program.
    Only modules that can be imported from the current directory
    will be included. This program does not run the code, so import statements
    in if/else or try/except blocks will always be included.
    """
    importedItems = []
    print("LOGGING:Checking file for imports:",toCheck)
    with open(toCheck, 'r',encoding="utf8") as pyFile:
        for line in pyFile:
            # ignore comments
            if "import " in line:
                if "from" not in line:
                    line_=line.replace("import ","").replace("\n","").split(" ")
                    final_import=[]
                    for elem in line_:
                        if elem=="as":
                            break
                        else:
                            if "," in elem:
                                tmp=elem.split(",")
                                for elem_ in tmp:
                                    if "." in elem_:
                                        tpath = handle_dotted_imports(line_)
                                        final_import.append(tpath)
                                    else:
                                        final_import.append([elem_])
                            else:
                                if "." in elem:
                                    tpath = handle_dotted_imports(elem)
                                    final_import.append(tpath)
                                else:
                                    final_import.append([elem])

                    for elem in final_import:
                        importedItems.append(elem)
                else:
                    line_ = line.replace("from ", "").replace("\n ", "")
                    line_=line_.split("import")[0].replace(" ","")
                    if "." in line_:
                        tpath=handle_dotted_imports(line_)
                        importedItems.append(tpath)
                    else:
                        importedItems.append([line_])
    return importedItems

def handle_file_with_imports(imports,files):
    repository_files_imp=[]
    for elem in imports:
        res=check_if_file_in_files(elem,files)
        if res==True:
            repository_files_imp.append(elem)
    return repository_files_imp

def check_if_file_in_files(elem,files):
    for tmp in files:
        if len(elem)==1:
            if elem[0]==tmp[-1]:
                return True
        else:
            if elem in files:
                return True
    print("ERROR:Did not found file ",elem," in ",files)
    return False

def check_lists(pathList,suffix,produced_files,log=0):
    retPathList=[]
    pathList=list(pathList)
    for prFile in produced_files:
        for file in pathList:
            #if log ==1:
                #print("check_lists:::::prFile=",prFile,"  file=",file)
            if (prFile+suffix) in str(file):
                retPathList.append(file)
                break
    return retPathList

#Beginning of thesis
with open('github/github.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    cnt___=0
    for url in csv_reader:
        repository_path=github.get_github_repository(url[0])
        code_repository=github.folder+github.dirName+"/"+repository_path
        windows=github.folder + github.dirName + "\\" + repository_path+"\\"
        print("LOGGING:About to start processing repository into ",code_repository)
        main_files=[]
        function_files=[]
        file_import_dict={}
        #File counter is needed in order to insert layers into annetto with distinct names.
        #For example two networks may have a FC layer with name add,the insertion should be able to give them different names
        #add234 and add124.For each item inserted it increases the counter,thus each item inserted gets a new number added
        #to the original name.
        file_counter =0
        with open("counter.txt", "r") as ins:
            for line in ins:
                file_counter=int(line.replace("\n",""))
        if True==True:
            function_files = []
            found_network=False
            from pathlib import Path
            pathlist = Path(code_repository).glob('**/*.py')
            files=[]
            trans={}
            for path in pathlist:
                files.append(str(path).replace(".py","").split(windows)[1].split("\\"))
                trans["/".join(files[-1])]=str(path)
            pathlist = Path(code_repository).glob('**/*.py')
            skip=False
            for path in pathlist:
                #Βecause path is object not string.
                path_in_str = str(path)
                imported=find_imports(path_in_str)
                import ntpath
                subdir=ntpath.dirname(path_in_str)
                #Identify the structure of the code ,by finding out possible main files(files do not included anywhere)
                #and function files.
                print("\nLOGGING:Begin searching for ",path_in_str,"with imported ",imported,"\n")
                repo_files_imp=handle_file_with_imports(imported,files)
                import_paths=[]
                for imports in repo_files_imp:
                    t_key="/".join(imports)
                    if len(imports)==1:
                        for key in trans.keys():
                            if str(key).endswith(t_key):
                                t_key=key
                                break
                    function_files.append(str(trans[t_key]))
                    import_paths.append(str(trans[t_key]))
                print("LOGGING:Imported files are ",import_paths)
                file_import_dict[str(path)] = import_paths
            pathlist = Path(code_repository).glob('**/*.py')
            for path in pathlist:
                #Iterate over all posible main files.Disregard included files and __init__ files
                if str(path) not in function_files and "__init__" not in str(path):
                    used_files=file_import_dict[str(path)]
                    project_structure=[]
                    while used_files!=[]:
                        tmp=[]
                        for elem_ in used_files:
                            if elem_!=[]:
                                if elem_ not in project_structure:
                                    project_structure.append(elem_)
                                    if file_import_dict[elem_]!=[]:
                                        for x in file_import_dict[elem_]:
                                            if x!=[]:
                                                tmp.append(x)
                            used_files=tmp
                    (result,found_network,handler_entities,produced_files)=handle_main_file(str(path),subdir,file_counter)
                    produced_files=[m for m in produced_files if "sEssIOn" in m]
                    print("LOGGING:Produced files are ",produced_files)
                    if found_network == True:
                        pathlistInfo = Path(code_repository).glob("**/*.info")
                        pathlistLine = Path(code_repository).glob("**/*.lines")
                        pathlistBatch = Path(code_repository).glob("**/*.batch")
                        pathListSession=Path(code_repository).glob("**/*.total_session")
                        timeList=[]
                        pathlistInfo=check_lists(pathlistInfo,'.info',produced_files)
                        pathlistLine = check_lists(pathlistLine, '.lines', produced_files)
                        pathListSession = check_lists(pathListSession, '.total_session', produced_files,1)
                        pathlistBatch = check_lists(pathlistBatch, '.batch', produced_files)
                        print("LOGGING:Batch files are ",pathlistBatch)
                        print("LOGGING:Session files are ", pathListSession)
                        for file in pathlistInfo:
                            timeList.append([os.path.getmtime(str(file)),str(file)])
                        timeList=sorted(timeList,key=lambda x:float(x[0]))
                        timeList=[x[1] for x in timeList]
                        pathlistInfo = Path(code_repository).glob("**/*.info")
                        session_dict=handle_lines_and_info.handle_lines_and_info(produced_files,pathlistInfo,pathlistLine,pathlistBatch,pathListSession)
                        import ntpath
                        session_dict=handle_lines_and_info.find_next_session_and_step(session_dict,[x.replace(".info","") for x in timeList])
                        session_dict=handle_lines_and_info.update_inner_epochs(session_dict)
                        temp_dict=session_dict.copy()
                        new_dict={}
                        for sess in session_dict.keys():
                            new_dict[sess.replace("\\","_").replace("/","_").replace(" ","_")]=session_dict[sess]
                            new_dict[sess.replace("\\","_").replace("/","_").replace(" ","_")].remove_unsupported_chars()
                            print("\n\n----------------------------------------------")
                            print("About to see info for ",sess)
                            new_dict[sess.replace("\\","_").replace("/","_").replace(" ","_")].print()
                            print("----------------------------------------------\n\n")
                        handler_entities.find_training(new_dict)
                        file_counter=tensorflow_parser.insert_in_annetto()
                        f = open("counter.txt", "w")
                        f.write(str(file_counter))
                        f.close()
                        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

            break