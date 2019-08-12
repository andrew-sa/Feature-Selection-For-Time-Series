import os
import sys

# import logging

from log_files.log import logger

LOGGER_EXTRA_OBJECT = {'caller_absolutepathname': os.path.abspath(__file__)}

def main():
    # logging.basicConfig(filename = './Logs/myapp.log', format = '%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s', level = logging.DEBUG)
    
    # logging.info('we')

    logger.info('I\'m the main', extra = LOGGER_EXTRA_OBJECT)
    
    filename = sys.argv[1]
    feature_number = int(sys.argv[2])
    
if __name__ == '__main__':
    main()