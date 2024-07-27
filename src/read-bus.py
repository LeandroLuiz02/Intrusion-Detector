from can import *
from utils import *
import numpy as np
import joblib
from tensorflow.keras.models import load_model

#NOT WORKING, THE MODEL DOES NOT SAVE PROPRIETLY
#PROBABLY THE MODEL IS NOT SAVING THE WEIGHTS

# Carregar o modelo treinado
model = load_model('idsmodel.h5')

# Carregar o escalador e usar os dados de treinamento para ajustá-lo
scaler = joblib.load('scaler.pkl')
X_train_load = np.load('X_train.npy')
X_train = scaler.fit_transform(X_train_load)

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

        bus = interface.Bus(channel=ch, interface=inter)
        filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=2, tolerance=0.03, enable_time=False)

        try:
                while True:
                        msg = CANMsgFromBus(bus.recv())
                        print(str(msg))
                        print("Filter test:")
                        filter_result = filter.test(msg)
                        print(filter.test(msg))

                        if filter_result == 'Normal':
                                # Extrair características da mensagem
                                features = np.array(extract_features(msg.message)).reshape(1, -1)
                                features_scaled = scaler.transform(features)

                                # Fazer a predição com o modelo treinado
                                y_pred_prob = model.predict(features_scaled)
                                threshold = 0.7
                                y_pred = (y_pred_prob > threshold).astype("int32")

                                if y_pred == 1:
                                        print('Intrusion detected!')
                                else:
                                        print('Normal message')
        except KeyboardInterrupt:
                print("interrupted by user")
        finally:
                bus.shutdown()

if __name__ == "__main__":
        main()
