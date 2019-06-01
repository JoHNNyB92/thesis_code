from github import github
import os
import sys
import tranform_tf_file
import tensorflow_parser


url="https://github.com/aymericdamien/TensorFlow-Examples.git"
repository_path=github.get_github_repository(url)
print(os.getcwd())
print("../git_repositories/"+repository_path+"/2")
for subdir, dirs, files in os.walk("../git_repositories/"+repository_path+"/TensorFlow-Examples/examples/2"):
#for subdir, dirs, files in os.walk("../git_repositories/stock-rnn-master"):
    for file in files:
        print(file)
        if file.endswith('.py'):
            total_path=os.path.join(subdir, file)
            print("Total path:",total_path)
            with open(total_path, encoding="utf8",errors='ignore') as myfile:
                enter=0
                if '.run(' in myfile.read():
                    enter=enter+1
                '''
                myfile.seek(0)
                if  ".train" in myfile.read():
                    enter+=0.5
                myfile.seek(0)
                if  "import tensorflow" in myfile.read():
                    enter+=0.5
                '''
                if enter>=1:
                    print("Found file :",file)
                    print("----------------------------------------------------------------------------")
                    path=os.getcwd()
                    batch_size=0
                    epoch=0
                    (pbtxt_file,batch_size,epoch,fileName)=tranform_tf_file.parse_file(total_path)
                    os.chdir(path)
                    print("----------------------------------------------------------------------------")
                    print("Finished executing file :",os.path.basename(total_path))
                    #pbtxt_file="..\_git_repositories\_topologies\_tensorflow\pbtxt\\"+os.path.basename(total_path)+".pbtxt"
                    print("Begin parsing file ", pbtxt_file)
                    tensorflow_parser.begin_parsing(os.path.basename(total_path),pbtxt_file,batch_size,epoch)
                    print("----------------------------------------------------------------------------")
                    print("Finished parsing of file ",os.path.basename(total_path))
                    #sys.exit()
