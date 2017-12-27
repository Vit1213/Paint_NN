import colorama

class text_color:
    HEADER = '\033[95m'
    BLUEMASSAG = '\033[94m'
    OK = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDs = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

colorama.init()

def get_WARNING_color_text():
    return text_color.WARNING + "WARNING!!!" + text_color.ENDs

def get_ERROR_color_text():
    return text_color.ERROR + "ERROR!!!" + text_color.ENDs

def get_OK_color_text():
    return text_color.OK + "OK" + text_color.ENDs