# aggregator/base.py

class AggregatorBase:
    def __init__(self, args):
        self.args = args

    def aggregate(self, local_updates, server):
        """
        local_updates: Danh sách dict từ các client, 
                       mỗi dict có dạng {"client_id": cid, "delta": <delta_param>}
        server: Tham chiếu tới Server (nếu cần đọc server.client_graph, server.sd, ...)
        
        Trả về: state_dict hoặc None tuỳ logic
        """
        raise NotImplementedError
