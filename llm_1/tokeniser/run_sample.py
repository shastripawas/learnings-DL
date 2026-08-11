from train import train_tokeniser
import pickle
from pathlib import Path
from infer import encode_text, decode_text

#text = Path("/home/shastri/TinyStories-train.txt").read_text(encoding="utf-8")
#chunk_size = 4 * 1024 * 1024  # 4 MB
enc_map = {}
spe_enc_map = {}
# with Path("../data/TinyStories-train.txt").open("r", encoding="utf-8") as f:
#     while True:
#         chunk = f.read(chunk_size)
#         if not chunk:
#             break

#         # Process this chunk
#         #process(chunk)
#         enc_map_temp, spe_enc_map_temp = train_tokeniser(chunk, 16384)
#         enc_map.update(enc_map_temp)
#         spe_enc_map.update(spe_enc_map_temp)


# with open("./trained/iter_1_main.pkl", "wb") as file:
#     pickle.dump(enc_map, file)

# with open("./trained/iter_1_main.pkl", "wb") as file:
#     pickle.dump(spe_enc_map, file)

with open('./trained/iter_1_main.pkl', 'rb') as file:
    enc_map = pickle.load(file)

with open('./trained/iter_1_spe.pkl', 'rb') as file:
    spe_enc_map = pickle.load(file)

text = input("Enter your text : ")
utf_e = list(map(int, text.encode("utf-8")))
print(f"Here is the utf : {utf_e}")
print(f"utf length : {len(utf_e)}")
enc = encode_text(text, enc_map, spe_enc_map)
print(f"Here is the encoding : {enc}")
print(f"utf length : {len(enc)}")
print(f"decoded encoding : {decode_text(enc,enc_map, spe_enc_map)}")