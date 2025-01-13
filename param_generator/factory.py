# param_generator/factory.py

from param_generator.hypernetwork import HyperNetworkParamGen
# from param_generator.ipg import ImplicitParamGen  # ví dụ

def get_param_generator(args):
    pg_type = args.get('param_gen', None)
    if pg_type == 'hypernetwork':
        return HyperNetworkParamGen(args)
    # elif pg_type == 'ipg':
    #     return ImplicitParamGen(args)
    else:
        return None
