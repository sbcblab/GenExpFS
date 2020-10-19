class Dataset:
    def __init__(self, name, data, classes, columns):
        self.name = name
        self.data = data
        self.classes = classes
        self.columns = columns

    def get(self):
        return self.data, self.classes, self.columns

    def get_instances(self):
        return self.data

    def get_classes(self):
        return self.classes

    def get_column_names(self):
        return self.columns
