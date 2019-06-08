from github import github
import os
import sys
import tranform_tf_file
import tensorflow_parser
import csv


def code_in_one_file(file):
    found_network=False
    result=""
    if file.endswith('.py'):
        total_path = os.path.join(subdir, file)
        with open(total_path, encoding="utf8", errors='ignore') as myfile:
            has_sess = False
            has_run=False
            is_main=False
            if '.run(' in myfile.read():
                has_run=True
            myfile.seek(0)
            if  'tf.Session' in myfile.read():
                has_sess = True
            myfile.seek(0)
            if '__main__' in myfile.read():
                is_main = True
            skip = ""
            print("FILE= ",file," hasRun=",has_run," HasSess=",has_sess," is main=",is_main)
            if (has_sess==True and has_run==True) or is_main==True:
                batch_size = 0
                epoch = 0
                path = os.getcwd()
                while "ERROR" not in result and "error" not in result and result != "success" and result!="batch_epoch_file_not_found":
                    (result, pbtxt_file, batch_size, epoch) = tranform_tf_file.parse_file(total_path, skip)
                    print("RESULT IS ",result)
                    if result == "batch":
                        print("ERROR:Unable to find batch number for ", file, ".Batch will be set as -1")
                        skip += result
                        batch_size = -1
                    os.chdir(path)

                if "error" in result:
                    print("ERROR:Error occured when executing the program ", file)
                else:
                    print("LOGGING:Finished executing file :", os.path.basename(total_path))
                    os.chdir(path)
                    print("----------------------------------------------------------------------------")
                    # Change directory in order to be appropriate for the folder that the pbtxt parser is located.
                    pbtxt_file = github.folder + github.dirName + pbtxt_file.split(github.dirName)[1]
                    # pbtxt_file="../git_repositories_temp\_tensorflow\pbtxt\\autoencoder.py.pbtxt"
                    # Windows OS
                    log_file = pbtxt_file.split("\\")[-1].replace(".py.pbtxt", "")
                    # Unix OS
                    log_file = log_file.split("/")[-1].replace(".py.pbtxt", "")
                    log_file = "../log/" + log_file
                    print("LOCO=",os.path.exists(pbtxt_file))
                    if os.path.exists(pbtxt_file)==False:
                        print("ERROR:There was an error with the creation of pbtxt file  ",log_file)
                    else:
                        print("LOGGING:Begin parsing pbtxt file ", pbtxt_file, " with anneto logging in ", log_file)
                        result = tensorflow_parser.begin_parsing(os.path.basename(total_path), pbtxt_file, batch_size,
                                                             epoch, log_file)
                        if result == "success":
                            found_network = True
                        print(
                            " ----------------------------------------------------------------------------------------------------------------------")
                        print("|Finished parsing of file ", os.path.basename(total_path), " Result:", result, "|")
                        print(
                            " ---------------------------------------------------------------------------------------------------------------------")
    return(result,found_network)

def code_in_multiple_files(file):
    print(file)

with open('github/github.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for url in csv_reader:
        print("LOGGING:Url for repository is ",url[0])
        repository_path=github.get_github_repository(url[0])
        code_repository=github.folder+github.dirName+"/"+repository_path
        print("LOGGING:About to start processing repository into ",code_repository)
        if "tutorials" in code_repository:
            found_network=False
            for subdir, dirs, files in os.walk(code_repository):
                for file in files:
                    (result,found_net)=code_in_one_file(file)
                    if found_net==False:
                        print("LOGGING:No network found in ",file)
                    if found_network==False:
                        found_network=found_net



