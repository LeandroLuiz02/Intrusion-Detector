class CommunicationMatrix():
    def __init__(self, file_path):
        import json
        with open(file_path, 'r') as f:
            self.matrix = json.load(f)

    def __str__(self):
        # TODO: Implement this
        return ""

class CANMessage():
    def __init__(self, time_stamp, id, payload, p_len, label = None):
        self.time_stamp = time_stamp
        self.id = id
        self.payload = payload
        self.p_len = p_len
        self.label = label

    def __str__(self):
        return f"\
    ===============\n\
    ID: {self.id}\n\
    Time Stamp: {self.time_stamp}\n\
    Payload Length: {self.p_len}\n\
    Payload: {self.payload}\n\
    Label: {self.label if self.label is not None else 'Unknown'}"

def CANMsgFromline(line : str):
    from math import ceil
    if line:
        words = line.split(' ')
        time_stamp = words[0][1:-1]
        id, _, payload = words[2].partition('#')
        label = ('Attack' if words[3] == 'T' else 'Normal') if len(words) == 4 else None
        return CANMessage(time_stamp, id, int(payload, 16), ceil(len(payload)/2), label)

class Filter():
    def __init__(self, comm_matrix : CommunicationMatrix, threshold = 3):
        self.comm_matrix = comm_matrix
        self.threshold = threshold
        self.cnt = 0

    def check_id_exists(self, msg : CANMessage):
        return msg.id in self.comm_matrix.matrix

    def check_payload_compatible(self, msg):
        return True # TODO: Implement this

    def check_is_in_time(self, msg):
        return True # TODO: Implement this

    def test(self, msg : CANMessage):
        check_properties = [
            self.check_id_exists(msg),
            self.check_payload_compatible(msg),
            self.check_is_in_time(msg)
        ]
        if all(check_properties):
            # Reset counter if message is normal
            self.cnt = 0
            return 'Normal'
        else:
            self.cnt += sum(1 for x in check_properties if x is False)
            return 'Attack' if self.cnt > self.threshold else 'Normal'
