file  = open("../attacks/DOS_ATCK.txt", 'r')
text = file.read()

class CANMessage():
    def __init__(self, time_stamp, id, payload, label):
        self.time_stamp = time_stamp
        self.id = id
        self.payload = payload
        self.label = label
    
    def __str__(self):
        return f'{self.time_stamp} {self.id} {self.payload} {self.label}'

ids = set()
for i, line in enumerate(text.split('\n')):
    v = line.split(' ')
    time_stamp = v[0][1:-1]
    id, _, payload = v[2].partition('#')
    label = 'Attack' if v[3] == 'T' else 'Normal'

    mg = CANMessage(time_stamp, id, payload, label)
    if label != 'Attack':
        ids |= {id}

print(ids)