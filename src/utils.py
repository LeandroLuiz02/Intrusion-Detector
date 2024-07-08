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
    def __init__(self, comm_matrix : CommunicationMatrix, threshold = 3, enable_time = True, tolerance = 0.04):
        self.comm_matrix = comm_matrix
        # Just some random big negative number to prevent the
        # first message from being classified as an attack
        self.prev_msg_time = {}
        for id in self.comm_matrix.matrix:
            self.prev_msg_time[id] = -0xFFFF
        # Store if the last two messages were attacks or not
        self.prev_msg_label = 'Normal'
        self.acc = 0
        self.enable_time = enable_time
        self.threshold = threshold
        self.tolerance = tolerance

    def check_id_exists(self, msg : CANMessage):
        return msg.id in self.comm_matrix.matrix

    def check_payload_compatible(self, msg):
        def get_val_at_bit_interval(signal, start, end):
            mask = 0xFFFFFFFFFFFFFFFF # 64 bits, maximum length is 8 bytes
            # Bring the start bit to the beginning
            signal = signal >> start
            # Create a mask with the desired length and invert it's bits
            mask = (mask << (end - start)) ^ mask
            # Apply the mask to the signal
            return signal & mask

        def calculate_payload_vals(payload, id_info):
            s = 0 # Start index
            e = 1 # End index
            inter = 'bit_interval' # Just to make the code more readable
            return [
              # Get the value at the bit interval, apply the offset and factor
              (get_val_at_bit_interval(payload, signal[inter][s], signal[inter][e]+1) + signal['offset']) * signal['factor']
              for signal in id_info['signals']
            ]

        id_info = self.comm_matrix.matrix[msg.id]

        # Check if the payload length is compatible
        is_len_compatible = id_info['length'] >= msg.p_len
        if not is_len_compatible:
            return False

        # Check if the signal values are compatible, all signals must be in the range
        signal_values = calculate_payload_vals(msg.payload, id_info)
        return all([
            (True if (signal_values[i] >= signal['min'] and signal_values[i] <= signal['max']) else False)
            for i, signal in enumerate(id_info['signals']) 
        ])

    def check_is_in_time(self, msg):
        time = float(msg.time_stamp)
        is_in_time = True
        # Only check if the previous message was normal, possible attack coming
        if self.prev_msg_label == 'Normal': 
            is_in_time = (time - self.prev_msg_time[msg.id]) > self.tolerance
        return is_in_time

    def test(self, msg : CANMessage):
        is_valid_id = self.check_id_exists(msg)
        check_properties = [ is_valid_id ]

        # Only make sense to check the payload
        # and the time of some message if the id is valid
        if is_valid_id:
            check_properties.append(self.check_payload_compatible(msg))
            # Only uses the time as a property if the time check is enabled
            if self.enable_time:
                check_properties.append(self.check_is_in_time(msg))

        if all(check_properties):
            # Reset acc if message is normal
            self.acc = 0
            self.prev_msg_label = 'Normal'
        else:
            # Accumulate errors to classify the message
            self.acc += sum(1 for x in check_properties if not x)
            self.prev_msg_label = 'Attack' if self.acc > self.threshold else 'Normal'

        # Saving the last normal message of this id
        if self.prev_msg_label == 'Normal':
            self.prev_msg_time[msg.id] = float(msg.time_stamp)

        return self.prev_msg_label
