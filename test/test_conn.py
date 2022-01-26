
import os
from fabric import Connection
from ray_tpu import check_tpu

name = "inf1"
zone = "us-central1-b"
info = check_tpu(name, zone)
i = info["networkEndpoints"][0]
# ip = i["ipAddress"]
ip = '34.121.189.187'
conn = Connection(ip,
     connect_kwargs={
     "key_filename": [os.path.expanduser('~/.ssh/google_compute_engine')], })
