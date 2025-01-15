# aggregator/factory.py

from aggregator.base import AggregatorBase
from aggregator.sheaf import SheafDiffusionAggregator

def get_aggregator(args):
    agg_type = args.get('aggregator','sheaf')
    if agg_type=='sheaf':
        return SheafDiffusionAggregator(args)
    else:
        raise ValueError(f'Unknown aggregator: {agg_type}')
