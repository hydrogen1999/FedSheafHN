# aggregator/factory.py

from aggregator.sheaf import SheafDiffusionAggregator
# from aggregator.hodge import HodgeAggregator  # v.v.

def get_aggregator(args):
    agg_type = args.get('aggregator', 'sheaf')
    if agg_type == 'sheaf':
        return SheafDiffusionAggregator(args)
    # elif agg_type == 'hodge':
    #    return HodgeAggregator(args)
    else:
        raise ValueError(f"Unknown aggregator: {agg_type}")
