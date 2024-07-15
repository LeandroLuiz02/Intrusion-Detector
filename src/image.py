from PIL import Image
import numpy as np

def hex_to_binary(hex_str):
    # Remove o prefixo "0x" e converte para inteiro
    hex_int = int(hex_str, 16)
    # Converte para binário e remove o prefixo "0b"
    binary_str = bin(hex_int)[2:]
    # Garante que a string binária tenha 12 bits, preenchendo com zeros à esquerda se necessário
    string = binary_str.zfill(12)
    return list(map(int, string))  # Converte cada caractere da string para inteiro

# Extracting every byte from a payload
def payload_convert_each_byte(payload):
    values = []
    for i in range(4):
        values.append(((payload >> (8 * i)) & 0xFF)/255.0)
    return values

def payload_to_binary(payload): return list(map(int, bin(payload)[2:].zfill(32)[:32]))

def create_can_image(can_messages, mirror=False):
    """
    Cria uma imagem a partir de mensagens CAN onde cada linha representa uma mensagem
    e cada quadrado representa um bit (1 -> branco, 0 -> preto).
    
    Args:
    - can_messages (list of str): Lista de strings, cada uma representando um ID CAN em hexadecimal.
    - square_size (int): Tamanho do lado de cada quadrado em pixels.
    
    Returns:
    - PIL.Image: Imagem gerada.
    """
    # Converte os IDs CAN de hexadecimal para binário
    can_messages_binary = [hex_to_binary(msg.id) + payload_convert_each_byte(msg.payload) for msg in can_messages]
    # Adiciona a mesma mensagem invertida para criar um efeito de espelhamento
    if mirror:
        can_messages_binary += can_messages_binary[::-1]
    
    img = Image.fromarray(np.array(can_messages_binary) , 'L')
    return img
    
    # Quantidade de mensagens e bits por mensagem
    num_messages = len(can_messages_binary)
    num_bits = len(can_messages_binary[0])
    
    # Dimensões da imagem
    width = num_bits * square_size
    height = num_messages * square_size
    
    # Criar uma imagem em branco
    # img = Image.new('RGB', (width, height), 'white')
    # draw = ImageDraw.Draw(img)
    # 
    # # Preencher a imagem com quadrados conforme os bits
    # for row, message in enumerate(can_messages_binary):
    #     for col, bit in enumerate(message):
    #         top_left = (col * square_size, row * square_size)
    #         bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
    #         color = 'white' if bit == '1' else 'black'
    #         draw.rectangle([top_left, bottom_right], fill=color)
    
    return img

# Exemplo de uso
# can_messages = [
#     "0x38B",
#     "0x15E",
#     "0x7F1",
#     "0x00C",
#     "0xA74",
#     "0x3D3",
#     "0x720",
#     "0x10F",
#     "0x881",
#     "0x54C",
#     "0xD2A",
#     "0x0B9",
# ]
#
# img = create_can_image(can_messages)
# img.show()  # Isso exibirá a imagem gerada
# img.save("can_messages.png")  # Isso salvará a imagem em um arquivo
