from can import *
from utils import *

def CANMsgFromBus(msg):
        id = (hex(msg.arbitration_id))[2:].upper()
        pad_id = ((3-len(id)) * '0') + id if len(id) != 3 else id
        t = str(msg.timestamp)
        d_len = msg.dlc
        d = int.from_bytes(msg.data)
        return CANMessage(t, pad_id, d, d_len)

def main():
        inter = "socketcan"
        ch = "can0"

        bus = interface.Bus(channel=ch, interface=inter)
        filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=2, tolerance=0.03, enable_time=False)

        try:
                while True:
                        msg = CANMsgFromBus(bus.recv())
                        print(str(msg))
                        print("Filter test:")
                        print(filter.test(msg))

        except KeyboardInterrupt:
                print("interrupted by user")
        finally:
                bus.shutdown()

if __name__ == "__main__":
        main()
