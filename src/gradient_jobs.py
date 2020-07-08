
import subprocess
import pandas as pd
from utils import get_max_version
import glob
import os
import shutil

def get_jobs():
    print("Fetching Jobs")
    def has_data(job_id):
        process = subprocess.Popen(['gradient','jobs','artifacts','list','--id', job_id],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return not("No data found" in stdout.decode("utf-8").split('\r')[-2])

    process = subprocess.Popen(['gradient','jobs','list','--project', 'Job Builder'],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    x=stdout.decode("utf-8") 
    x=x.split("\n")
    jobs=x[3:-1]
    jobs=[[j.strip() for j in  i.split("|")][1:-1] for i in jobs]
    jobs=pd.DataFrame(jobs,columns=["id","Name","from","location","device","time"])
    jobs["has_data"]=jobs["id"].apply(has_data)
    return jobs

def download_jobs(jobs,dir):
    print("Downloading Jobs")
    def download_job(job_id,download_dir):
        process = subprocess.Popen(['gradient','jobs','artifacts','download','--id', job_id ,"--destinationDir" ,download_dir],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    to_download=jobs[jobs["has_data"]]
    job_list=to_download["id"]
    for job in job_list:
        download_job(job,f"{dir}/{job}")
    print("Jobs Downloaded")

def copy_to_location(transient_dir):
    jobs = glob.glob(f"{transient_dir}*")
    for job_dir in jobs:
        copy_to_saved(job_dir)
    
def copy_to_saved(job_dir):
    try:
        job_path = glob.glob(f"{job_dir}/*/")[0]
        model_type = job_path.split("/")[-2]
        print(model_type)
        _ver = get_max_version(model_type)
        version = max(_ver)+1 if _ver!=[] else 0
        model_path = f"./src/saved_models/{model_type}/{str(version)}"
        os.makedirs(model_path)
        print(model_path)
        data_files = glob.glob(job_dir+f"/{model_type}/*/*")
        for file_name in data_files:
            shutil.copy(file_name,'%s/%s' % (model_path, file_name.split('/')[-1]))
    except:
        print(f"Failed to load {job_dir}")
    
if __name__=="__main__":
    transient_dir = "./gradient/"
    jobs=get_jobs()
    download_jobs(jobs,transient_dir)
    copy_to_location(transient_dir)