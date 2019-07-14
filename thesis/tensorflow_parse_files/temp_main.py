from github import github
import os
import sys
import tranform_tf_file
import tensorflow_parser
import handle_lines_and_info
import csv
from modulefinder import ModuleFinder

def code_in_one_file(file,subdir):
    found_network=False
    total_path = os.path.join(subdir, file)
    result=""
    handler_entities=""
    if str(file).endswith('.py'):
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
            print("FILE= ",file," hasRun=",has_run," HasSess=",has_sess," is main=",is_main)

            if (has_sess==True and has_run==True) or is_main==True or (has_def_main==True and has_tf_app_run==True) or has_interactive==True:
                path = os.getcwd()
                tf_run_app=False
                if (has_def_main==True and has_tf_app_run==True):
                    tf_run_app=True
                result= tranform_tf_file.parse_file(file,tf_run_app,project_structure)
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
                    print("PBTXT=",pbtxt_file)
                    import sys
                    sys.exit()
                    #pbtxt_file = github.folder + github.dirName + pbtxt_file.split(github.dirName)[1]

                    # Windows OS
                    log_file = pbtxt_file.split("\\")[-1].replace(".py.pbtxt", "")
                    # Unix OS
                    log_file = log_file.split("/")[-1].replace(".py.pbtxt", "")
                    log_file = "../log/" + log_file
                    if os.path.exists(pbtxt_file)==False:
                        print("ERROR:There was an error with the creation of pbtxt file  ",pbtxt_file)
                    else:
                        print("LOGGING:Begin parsing pbtxt file ", pbtxt_file, " with anneto logging in ", log_file)
                        (result,handler_entities)= tensorflow_parser.begin_parsing(os.path.basename(subdir), pbtxt_file,log_file)
                        if result == "success":
                            found_network = True
                        print(
                            " ----------------------------------------------------------------------------------------------------------------------")
                        print("|Finished parsing of file ", os.path.basename(total_path), " Result:", result, "|")

                        print(
                            " ---------------------------------------------------------------------------------------------------------------------")
    return(result,found_network,handler_entities)

def code_in_multiple_files(file):
    print(file)

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
    print("elelele")
    with open(toCheck, 'r') as pyFile:
        for line in pyFile:
            # ignore comments
            if "import " in line:
                if "from" not in line:
                    #print("ELEOR=",line)
                    line_=line.replace("import ","").replace("\n","").split(" ")
                    print("\n\n\n\n\n\nLINE_=",line_)
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
                                    print(final_import[-1])

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
    print("Locoooooo=",elem," is ",files)
    for tmp in files:
        if len(elem)==1:
            print("hooray1=",elem," tmp1=",tmp[-1])
            if elem[0]==tmp[-1]:
                print("Found file ",elem)
                return True
        else:
            if elem in files:
                print("Found file ", elem)
                return True
    print("Did not found file=",elem," in ",files)
    return False

with open('github/github.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for url in csv_reader:
        #print("LOGGING:Url for repository is ",url[0])
        repository_path=github.get_github_repository(url[0])
        code_repository=github.folder+github.dirName+"/"+repository_path
        windows=github.folder + github.dirName + "\\" + repository_path+"\\"
        print("LOGGING:About to start processing repository into ",code_repository)
        main_files=[]
        function_files=[]
        file_import_dict={}
        if "simple_file" in code_repository:
            function_files = []
            found_network=False
            from pathlib import Path
            pathlist = Path(code_repository).glob('**/*.py')
            files=[]
            trans={}
            for path in pathlist:
                print(str(path))
                files.append(str(path).replace(".py","").split(windows)[1].split("\\"))
                trans["/".join(files[-1])]=str(path)
                #file_import_dict[str(path)].append([str(path).split(windows)[1].split("\\")])
            print(files)
            pathlist = Path(code_repository).glob('**/*.py')

            for path in pathlist:
                # because path is object not string
                path_in_str = str(path)
                imported=find_imports(path_in_str)
                import ntpath
                subdir=ntpath.dirname(path_in_str)
                print("\n\n Begin searching for ",path_in_str,"with imported ",imported)
                repo_files_imp=handle_file_with_imports(imported,files)
                print("Found imports ", repo_files_imp)
                #file_import_dict["/".join(path_in_str)+".py"]
                import_paths=[]
                for imports in repo_files_imp:
                    t_key="/".join(imports)
                    if len(imports)==1:
                        for key in trans.keys():
                            print(key)
                            if str(key).endswith(t_key):
                                t_key=key
                                break
                    function_files.append(str(trans[t_key]))
                    import_paths.append(str(trans[t_key]))
                print("Path is = ",import_paths)
                file_import_dict[str(path)] = import_paths
                print("file_import_dict[",str(path),"]=",file_import_dict[str(path)])

            pathlist = Path(code_repository).glob('**/*.py')
            for path in pathlist:
                if str(path) not in function_files and "__init__" not in str(path):
                    used_files=file_import_dict[str(path)]
                    print(used_files)
                    project_structure=[]
                    while used_files!=[]:
                        tmp=[]
                        for elem_ in used_files:
                            if elem_!=[]:
                                if elem_ not in project_structure:
                                    print("used file=", elem_)
                                    project_structure.append(elem_)
                                    if file_import_dict[elem_]!=[]:
                                        for x in file_import_dict[elem_]:
                                            if x!=[]:
                                                tmp.append(x)
                            used_files=tmp

                    print("Files include in main file ",path," are ",project_structure)
                    (result,found_network,handler_entities)=code_in_one_file(str(path),subdir)
                    if found_network == True:
                        pathlistInfo = Path(code_repository).glob("**/*.info")
                        pathlistLine = Path(code_repository).glob("**/*.lines")
                        timeList=[]
                        for file in pathlistInfo:
                            timeList.append([os.path.getmtime(str(file)),str(file)])
                        timeList=sorted(timeList,key=lambda x:float(x[0]))
                        timeList=[x[1] for x in timeList]
                        pathlistInfo = Path(code_repository).glob("**/*.info")
                        print("pathListInfo=",pathlistInfo)
                        files = [
                                 "_simple_file_temporary_05_basic_convnet.py__sEssIOn_[2]_1",
                                 "_simple_file_temporary_05_basic_convnet.py__sEssIOn_[2]_co_train_2",
                                 ]

                        file_dict=handle_lines_and_info.handle_lines_and_info(files,pathlistInfo,pathlistLine)
                        print("\n\n\n\n\n\n\n\n\n",file_dict)
                        for d in file_dict:
                            print('ELELELELE=',file_dict[d].epoch)
                        import ntpath
                        for ind,elem in enumerate(timeList):
                            file_dict[ntpath.basename(elem).replace(".info","")].next_file=ntpath.basename(timeList[ind+1]).replace(".info","")
                            if ind+2==len(timeList):
                                break
                        for ind,elem in enumerate(timeList):
                            print(elem.replace(".info",""),"->",file_dict[ntpath.basename(elem).replace(".info","")].next_file)
                        handler_entities.find_training(file_dict)
            sys.exit()



