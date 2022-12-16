import os

from configparser import SafeConfigParser

config = os.getcwd() + '/config/parameter.ini'
if not os.path.exists(os.getcwd() + '/config/parameter.ini'):
    config = os.path.dirname(os.getcwd() + '/config/parameter.ini')

def get_config():
    parser = SafeConfigParser()
    parser.read(config)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)