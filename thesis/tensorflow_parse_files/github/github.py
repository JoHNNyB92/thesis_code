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
dirName = "../git_repositories_temp"

def create_folder(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("LOGGING:Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("LOGGING:Directory " , dirName ,  " already exists")


def get_github_repository(url):
    create_folder(dirName)
    repository_folder=os.path.basename(url)
    repository_path=dirName+"/"+repository_folder
    create_folder(repository_path)
    if len(os.listdir(repository_path)) == 0:
        git.Git(repository_path).clone(url)
        print("LOGGING:Created folder ",repository_path,"for repository ", url)
    else:
        print("ERROR:Folder ",repository_path," already contains a repository.")
    return os.path.basename(repository_folder)