# param_generator/base.py

class ParamGenBase:
    def __init__(self, args):
        self.args = args

    def prepare_params(self, server, clients, selected_client_ids):
        """
        Forward HN => generate param => store vào sd[client_id]['generated model params'].
        """
        raise NotImplementedError

    def backprop_hn(self, server, local_updates):
        """
        Backprop HN từ local updates => update HN.
        """
        pass  # optional