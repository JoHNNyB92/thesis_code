from github import github
import os
import sys
import tranform_tf_file
import tensorflow_parser
import csv

with open('github/github.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for url in csv_reader:
        print("LOGGING:Url for repository is ",url[0])
        repository_path=github.get_github_repository(url[0])
        code_repository=github.dirName+"/"+repository_path
        print("LOGGING:About to start processing repository into ",code_repository)
        for subdir, dirs, files in os.walk(code_repository):
            for file in files:
                if file.endswith('.py'):
                    total_path=os.path.join(subdir, file)
                    with open(total_path, encoding="utf8",errors='ignore') as myfile:
                        enter=0
                        if '.run(' in myfile.read():
                            enter=enter+1
                        result=""
                        skip=""
                        if enter >= 1:
                            batch_size = 0
                            epoch = 0
                            path = os.getcwd()
                            while result!="error" and result!="success":
                                wd=os.getcwd()
                                (result,pbtxt_file,batch_size,epoch)=tranform_tf_file.parse_file(total_path,skip)
                                if result=="batch":
                                    print("LOGGING:Unable to find batch number for ", file,".Batch will be set as -1")
                                    skip+=result
                                    batch_size =-1
                                os.chdir(path)
                            print("LOGGING:Finished executing file :",os.path.basename(total_path))
                            os.chdir(path)
                            print("----------------------------------------------------------------------------")
                            #Change directory in order to be appropriate for the folder that the pbtxt parser is located.
                            pbtxt_file=github.dirName+pbtxt_file.split(github.dirName)[1]
                            print("LOGGING:Begin parsing pbtxt file ", pbtxt_file)
                            result=tensorflow_parser.begin_parsing(os.path.basename(total_path),pbtxt_file,batch_size,epoch)
                            print(" ----------------------------------------------------------------------------------------------------------------------")
                            print("|Finished parsing of file ", os.path.basename(total_path), " Result:", result,"|")
                            print( " ---------------------------------------------------------------------------------------------------------------------")