import os
import requests
import json
import git
api_url = 'https://api.github.com'
#search_url="https://api.github.com/search/repositories?q=simple+neural+networks+tensorflow+language:python&sort=stars&order=desc"
proxy_url='https://10.144.1.10:8080'
proxies = {
  'https': proxy_url
}
dirName = "git_repositories"

def create_folder(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")


def get_github_repository(url):
    '''
    CASE OF SEARCH FOR RANDOM GITHUB TENSORFLOW REPOSITORIES
    r=requests.get(url)#, proxies=proxies)
    jsons=json.loads(r.content.decode("utf-8"))
    print(jsons.keys())
    repo_url=jsons["items"][0]["clone_url"]
    '''
    dir_name="../" + dirName
    create_folder(dir_name)
    repository_folder=os.path.basename(url)
    repository_path=dir_name+"/"+repository_folder
    create_folder(repository_path)
    if len(os.listdir(repository_path)) == 0:
        git.Git(repository_path).clone(url)
        print("Created folder ",repository_path,"for repository ", url)
    else:
        print("Folder ",repository_path," already contains a repository.")
    return os.path.basename(repository_folder)
    '''
    for subdir, dirs, files in os.walk("git_repositories/IITG-Captcha-Solver-OpenCV-TensorFlow.git"):
        for file in files:
            if file.endswith('.py2'):
                total_path=os.path.join(subdir, file)
                with open(total_path) as myfile:
                    #if '.run' in myfile.read():
                    if '.fit' in myfile.read():
                        print(total_path)
                        parse.parse_file(total_path)
    '''