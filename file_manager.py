__all__ = ['FileManager']


class FileManager(object):
    def __init__(self, filename, buffer_size=4096, overwrite=True):
        self.filename = filename
        self.buffer_size = buffer_size
        self.overwrite = overwrite
        mode = 'w' if overwrite else 'a'
        self._file = open(filename, mode=mode, buffering=buffer_size)
    
    def write_line(self, line):
        try:
            self._file.write(line + '\n')
        except ValueError:
            self._file = open(self.filename, mode='a', buffering=self.buffer_size)
            self._file.write(line + '\n')

    def close(self):
        self._file.close()
