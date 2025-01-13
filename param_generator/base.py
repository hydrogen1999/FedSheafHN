# param_generator/base.py

class ParamGenBase:
    def __init__(self, args):
        self.args = args

    def prepare_params(self, server, clients, selected_client_ids):
        raise NotImplementedError
