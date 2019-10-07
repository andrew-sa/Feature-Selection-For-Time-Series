import os
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('extraction')
logscript_path = os.path.dirname(os.path.abspath(__file__))
parent_logscript_path = os.path.split(logscript_path)[0]
pathname = parent_logscript_path + '/LogFiles' + '/extraction.log'
file_handler = logging.FileHandler(pathname)
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s \t\t\t\t\t [%(caller_absolutepathname)s]')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)