import os
import git

'''
This is a file that performs the download of the repository that contains the neural network.
If the folder is created,it does not re-download the repository.
'''

api_url = 'https://api.github.com'
proxy_url='https://10.144.1.10:8080'
proxies = {
  'https': proxy_url
}
dirName = "git_repositories_temp"
folder="..\\"

def create_folder(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("LOGGING:Directory " , folder+dirName ,  " Created ")
    except FileExistsError:
        print("LOGGING:Directory " , folder+dirName ,  " already exists")


def get_github_repository(url):
    create_folder(folder+dirName)
    repository_folder=os.path.basename(url)
    repository_path=folder+dirName+"/"+repository_folder
    create_folder(repository_path)
    if len(os.listdir(repository_path)) == 0:
        git.Git(repository_path).clone(url)
        print("LOGGING:Created folder ",repository_path,"for repository ", url)
    else:
        print("ERROR:Folder ",repository_path," already contains a repository.")
    return os.path.basename(repository_folder)