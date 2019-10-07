import os
import logging

'''
# print(os.path.dirname(os.path.abspath(__file__)))

# We don't use caller_absolutepathname because the feature selection of tsfresh causes an error when it calls logger beacause it hasn't the variable caller_absolutepathname
# logging.basicConfig(filename = os.path.dirname(os.path.abspath(__file__)) + '/app.log', format = '%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s \t\t\t\t\t [%(caller_absolutepathname)s]', level = logging.INFO)

# We use this configuration
logging.basicConfig(filename = os.path.dirname(os.path.abspath(__file__)) + '/app.log', format = '%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s - %(funcName)s()] - %(message)s', level = logging.INFO)

logger = logging.getLogger('root')
'''

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('app')
# pathname = os.path.dirname(os.path.abspath(__file__)) + '/app.log'
logscript_path = os.path.dirname(os.path.abspath(__file__))
parent_logscript_path = os.path.split(logscript_path)[0]
pathname = parent_logscript_path + '/LogFiles' + '/app.log'
file_handler = logging.FileHandler(pathname)
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s \t\t\t\t\t [%(caller_absolutepathname)s]')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)