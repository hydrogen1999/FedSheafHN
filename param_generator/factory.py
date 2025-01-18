# param_generator/factory.py

from param_generator.hypernetwork import HyperNetworkParamGen

def get_param_generator(args):
    pg_type = args.get('param_gen', None)
    if pg_type == 'hypernetwork':
        return HyperNetworkParamGen(args)
    else:
        # None => không dùng param generator
        return None
