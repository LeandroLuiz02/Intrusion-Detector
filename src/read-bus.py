from can import *
from utils import *
import numpy as np
import pandas as pd
import re
import joblib
from tensorflow.keras.models import load_model

def extract_features(message):
    parts = re.split('[#]', message)
    id_hex = parts[0]
    data_hex = parts[1] if len(parts) > 1 else ''

    id_int = int(id_hex, 16)
    data_bytes = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]

    data_bytes += [0] * (8 - len(data_bytes))

    return [id_int] + data_bytes

# Processa a mensagem CAN recebida e retorna um DataFrame
def process_single_message(msg):
        id = (hex(msg.arbitration_id))[2:].upper()
        pad_id = ((3-len(id)) * '0') + id if len(id) != 3 else id
        d = ''.join(format(byte, '02x') for byte in msg.data).upper()
        message = f'{pad_id}#{d}'
        data = pd.DataFrame({'message': [message]})
        return data

def CANMsgFromBus(msg):
        id = (hex(msg.arbitration_id))[2:].upper()
        pad_id = ((3-len(id)) * '0') + id if len(id) != 3 else id
        t = str(msg.timestamp)
        d_len = msg.dlc
        d = ''.join(format(byte, '02x') for byte in msg.data).upper()
        message = f'{pad_id}#{d}'
        return CANMessage(t, pad_id, d, d_len, message=message)

def main():
        inter = "socketcan"
        ch = "can0"

        # Carregar o modelo treinado
        model = load_model('idsmodel.h5')
        scaler = joblib.load('scaler.pkl')
        bus = interface.Bus(channel=ch, interface=inter)
        filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=3, tolerance=0.03, enable_time=False)

        try:
                while True:
                        msg = CANMsgFromBus(bus.recv())
                        msgmlp = process_single_message(bus.recv())
                        print(str(msg))
                        print("Filter test:")
                        filter_result = filter.test(msg)
                        print(filter.test(msg))

                        if filter_result == 'Normal':
                                extract = extract_features(msgmlp['message'][0])
                                features = np.array(extract).reshape(1, -1)
                                features_scaled = scaler.transform(features)
                                y_pred_prob = model.predict(features_scaled)
        
                                thresholdmlp = 0.7
                                y_pred = (y_pred_prob > thresholdmlp).astype("int32")

                                if y_pred == 1:
                                        print('Intrusion detected!')
                                else:
                                        print('Normal message')
                        else:
                                print('Intrusion detected!')
        except KeyboardInterrupt:
                print("interrupted by user")
        finally:
                bus.shutdown()

if __name__ == "__main__":
        main()
