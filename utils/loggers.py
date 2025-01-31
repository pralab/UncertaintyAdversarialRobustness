from utils.utils import join
import logging
import os

def set_up_logger(root, cuda, kwargs):
    logger = __init_logger(root, common_logger_path=join(root, f"logger-cuda-{cuda}.log"))
    
    # Write kwargs on in exp_path/info.txt and then write also on logger before running the rest
    logger.debug('#'*40 + '\n' + '#'*40 + '\n' '#'*40)
    logger.debug('\n' + __write_kwargs(kwargs, root, 'info'))
    logger.debug('#'*40 + '\n' + '#'*40 + '\n' '#'*40)

    return logger


# Procedure to set a <fname>.log file in the directory specified by <root>
def __init_logger(root, fname='progress', common_logger_path=None):
    logger = logging.getLogger(fname)
    logger.setLevel(logging.DEBUG)

    logger.handlers = []
    # set the file on which to write the logging messages
    fh = logging.FileHandler(os.path.join(root, f'{fname}.log'))
    # formatter_file = logging.Formatter('%(asctime)s - %(message)s')
    # example of output format:
    # [02-12 12:48:17] /home/dangioni/UncertaintyQuantificationUnderAttack/main_attack.py:174} DEBUG - <message>
    formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if common_logger_path is not None:
        # date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
        # common_fh = logging.FileHandler(os.path.join(common_root, f'{date}_progress.log'))
        common_fh = logging.FileHandler(common_logger_path)
        common_formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                    '%m-%d %H:%M:%S')
        common_fh.setFormatter(common_formatter)
        logger.addHandler(common_fh)

    # Don't remember exactly, something like flushing the messages in real time in the logging file
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(fh)
    logger.addHandler(streamhandler)
    return logger


# If script parameters are collected as a dictionary, write them in a readable text format
def __write_kwargs(kwargs, dirname, fname):
    s = ''
    for k, v in kwargs.items():
        s += f"{k}: {v}\n"
    with open(join(dirname, f"{fname}.txt"), 'w') as f:
        f.write(s)
    return s