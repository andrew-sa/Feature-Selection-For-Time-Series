import os
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('rfd')
logscript_path = os.path.dirname(os.path.abspath(__file__))
parent_logscript_path = os.path.split(logscript_path)[0]
pathname = parent_logscript_path + '/LogFiles' + '/rfd.log'
file_handler = logging.FileHandler(pathname)
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)