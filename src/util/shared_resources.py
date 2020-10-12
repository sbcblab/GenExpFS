resources = None


class SharedResources:
    @staticmethod
    def set_resources(r):
        global resources
        resources = r

    @staticmethod
    def get():
        global resources
        return resources
