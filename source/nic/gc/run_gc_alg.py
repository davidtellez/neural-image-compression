import traceback

import click
import json
import os
from pathlib import Path
from random import randint
from time import sleep
from gcapi.gcapi import Client


def cli(input_path: str, output_dir: str, algorithm_title: str, upload_session_wait=600, job_wait=6000, token=None):
    input_file_path = input_path
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()
    job_executor = JobExecutor(algorithm_title=algorithm_title,
        input_file_path=input_file_path,output_dir=output_dir,
        upload_session_wait=upload_session_wait,job_wait=job_wait, token=token
    )
    try:
        return job_executor.run()
    except Exception as ex:
        # print(traceback.print_exc())
        raise ex

class JobExecutor(Client):
    def __init__(self,algorithm_title="",input_file_path="",output_dir="",
                 upload_session_wait=640,job_wait=7200,token=None):
        if token is None or len(token)==0:
            token = os.environ["GCTOKEN"]
        if token is None or len(token)==0:
            raise ValueError('requires a token parameter or setting the environment variable GCTOKEN')
        super(JobExecutor, self).__init__(token=token)
        self.algorithm_title = algorithm_title
        self.input_file_path = Path(input_file_path)
        self.output_dir = Path(output_dir)
        self.allowed_amount_of_time_upload_session = upload_session_wait
        self.allowed_amount_of_time_job = job_wait
        alg = [a for a in self.algorithms.page() if a["title"] == self.algorithm_title][0]
        self.algorithm_image_url = alg["latest_ready_image"]
        self.algorithm_slug = alg['slug']

    def submit_upload_session(self):
        if self.input_file_path.is_dir():
            files = [f.resolve() for f in self.input_file_path.iterdir()]
        else:
            files = [self.input_file_path.resolve()]
        upload_session = self.upload_cases(files=files, algorithm=self.algorithm_slug)

        return upload_session["pk"]

    def wait_until_upload_session_is_finished(self, upload_session_pk):
        """
        Waits until images are uploaded to grand challenge and a job is scheduled.

        Parameters
        ----------
        upload_session_pk : str
            Primary key of the submitted upload session

        Returns
        ----------
        list_of_uploaded_image_urls: list
            List of urls of the images that were uploaded by this upload session
        """
        num_retries = 0
        for _ in range(self.allowed_amount_of_time_upload_session):
            us = self.raw_image_upload_sessions.detail(upload_session_pk)
            if us["status"] == "Succeeded":
                list_of_uploaded_image_urls = us["image_set"]
                print('upload succeeded')
                break
            else:
                num_retries += 1
                sleep((2 ** num_retries) + (randint(0, 1000) / 1000))
                print('.', end='')
        else:
            raise TimeoutError(
                f"Upload session did not succeed within the allotted time of "
                f"{self.allowed_amount_of_time_upload_session} seconds"
            )
        if not list_of_uploaded_image_urls:
            raise IOError(
                f"Images were not uploaded. The upload session has {us['status'].lower()}"
            )
        sleep(3)#sometimes http-error if trying to access image-job to soon
        return list_of_uploaded_image_urls

    def wait_until_job_is_finished(self, job_url, image_name):
        """
        Waits until the algorithm has finished running and result is available.

        Parameters
        ----------
        job_url: str
            URL of the scheduled job

        Returns
        ----------
        result: json
            Result of the algorithm
        """
        json_result = None
        overlay_links = []
        num_retries = 0
        job_pk = job_url[:-1].split("/")[-1]
        print('waiting for job %s' % str(job_url))
        overlay_out_pathes = []
        for _ in range(self.allowed_amount_of_time_job):
            job = self.algorithm_jobs.detail(job_pk)
            if job["status"] == "Succeeded":
                print('job succeeded')
                # result_url = job["result"]
                outputs = job['outputs']
                for outp in outputs:
                    if outp['interface']['slug']=='results-json-file':
                        json_result = outp['value']
                    elif outp['interface']['slug']=='generic-overlay':
                        overlay_detail_link = outp['image']
                        detail_pk = overlay_detail_link[:-1].split("/")[-1]
                        # overly_pks.append(overlay_pk)
                        # overlay_links.append(overlay_link)
                        out_detail = self.images.detail(detail_pk)
                        overlay_link = out_detail['files'][0]['file']
                        overlay_links.append(overlay_link)
                        overlay_name = out_detail['name']
                        if image_name is not None:
                            overlay_name = image_name+'_'+overlay_name
                        print('writing %s' % overlay_name)
                        r = self.get(overlay_link)
                        out_path = self.output_dir/overlay_name
                        with open(str(out_path), "wb") as f:
                            f.write(r.content)
                        overlay_out_pathes.append(out_path)
                    else:
                        print('Ignoring unknown result', outp)
                break
            else:
                num_retries += 1
                sleep((2 ** num_retries) + (randint(0, 1000) / 1000))
                print('.', end='')
        else:
            raise TimeoutError(
                f"Job {job_url} did not succeed within the allotted time of "
                f"{self.allowed_amount_of_time_job} seconds"
            )

        if json_result is None:
            raise IOError(
                f"No result is available. The job has {job['status'].lower()}"
            )
        # result = self.algorithm_results.detail(result_url[:-1].split("/")[-1])
        # return result["output"]
        return json_result, overlay_out_pathes

    def get_job_url(self, uploaded_image_pk):
        job_url_list = self.images.detail(uploaded_image_pk)["job_set"]
        if not job_url_list:
            raise IOError("Job was not scheduled correctly")
        return job_url_list[0]

    def run(self):
        print('uploading...', end='')
        upload_session_pk = self.submit_upload_session()
        list_of_uploaded_image_urls = self.wait_until_upload_session_is_finished(upload_session_pk)
        for uploaded_image_url in list_of_uploaded_image_urls:
            uploaded_image_pk = uploaded_image_url[:-1].split("/")[-1]
            uploaded_image = self.images.detail(uploaded_image_pk)
            image_name = Path(uploaded_image["name"]).stem
            # image_name = Path(image_name).stem
            if image_name.endswith('_'):#gc appends an underscore to the name for some reason
                image_name = image_name[:-1]
            job_url = self.get_job_url(uploaded_image_pk)
            result, overlay_out_pathes = self.wait_until_job_is_finished(job_url, image_name)
            json_path = str(self.output_dir / f"result_{image_name}.json")
            with open(json_path, "w") as output_file:
                json.dump(result, output_file)
                print('%s saved' % str(json_path))
        print('Done!')
        return overlay_out_pathes
if __name__ == "__main__":
    cli()
