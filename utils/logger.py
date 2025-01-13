# utils/logger.py

from datetime import datetime

class Logger:
    def __init__(self, args, gpu_id=0, is_server=False):
        self.args = args
        self.gpu_id = gpu_id
        self.is_server = is_server
        self.c_id = -1

    def switch(self, c_id):
        self.c_id = c_id

    def print(self, message):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        msg = f"[{now}][{self.args['model']}][gpu:{self.gpu_id}]"
        msg += "[server]" if self.is_server else f"[client:{self.c_id}]"
        msg += f" {message}"
        print(msg)
