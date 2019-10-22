import sys
import os
import time


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        message = str(time.asctime()) + ' | ' + message.strip('\r').strip('\n') + "\r\n"
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger()
    print(453453)
    print(path)
    print(type)