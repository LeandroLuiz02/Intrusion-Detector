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

        opt = get_cmd_args()
        setseed(42)

        gan = Discriminator(opt)
        gan.load_state_dict(torch.load('./model.pth', map_location='cpu'))
        gan.eval()

        ids = IDS(
            filter=Filter(
                CommunicationMatrix('./communication_matrix.json'),
                window_size=opt.window_size,
                threshold=2,
                tolerance=2e-4,
                enable_time=False
            ),
            model=gan,
            opt=opt)

        try:
                print('Reading the bus...')
                print('Press Ctrl+C to exit')
                while True:
                        msg = CANMsgFromBus(bus.recv())
                        res = ids.test(msg)
                        if res is not None:
                                print(res)
                        # If loopback is activated, discard duplicate messages
                        if opt.loop:
                                bus.recv()

        except KeyboardInterrupt:
                print("interrupted by user")
        finally:
                bus.shutdown()

if __name__ == "__main__":
        main()
