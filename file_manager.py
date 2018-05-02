class FileManager():
    def __init__(self, filename, buffer_size=4096, overwrite=True):
        self.filename = filename
        self.buffer_size = buffer_size
        mode = 'w' if overwrite else 'a'
        self.file = open(filename, mode=mode, buffering=buffer_size)
    
    def write_line(self, line):
        try:
            self.file.write(line + '\n')
        except ValueError:
            self.file = open(self.filename, mode='a', buffering=self.buffer_size)
            self.file.write(line + '\n')

    def close(self):
        self.file.close()
