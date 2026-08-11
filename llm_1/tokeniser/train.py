import regex as re
from tqdm import tqdm
import pickle
from pathlib import Path
import argparse

def bpe(utf_lists, vocab_size, start = 256):
  enc_map = {}
  with tqdm(desc="Training Progress", unit="items") as pbar:
      while start<vocab_size:
          updated_lists = {}
          maxi=0
          pair_map = {}
          for utf_list in utf_lists:
            utf_list = list(utf_list)
            n = len(utf_list)
            i=0
            while i<n-1:
              if (utf_list[i], utf_list[i+1]) in pair_map:
                pair_map[(utf_list[i], utf_list[i+1])]+=utf_lists[tuple(utf_list)]
              else:
                pair_map.update({(utf_list[i], utf_list[i+1]):utf_lists[tuple(utf_list)]})
              maxi = max(maxi, pair_map[(utf_list[i], utf_list[i+1])])
              i+=1
          
          for utf_list in utf_lists:
            utf_list = list(utf_list)
            i=0
            n = len(utf_list)
            new_list = []
            while i<n:
              if i==n-1:
                new_list.append(utf_list[i])
                break
              if pair_map[(utf_list[i], utf_list[i+1])]==maxi:
                #print(f"entry : {(utf_list[i], utf_list[i+1])} pair_map : {pair_map[(utf_list[i], utf_list[i+1])]}")
                if (utf_list[i], utf_list[i+1]) not in enc_map:
                  enc_map[(utf_list[i], utf_list[i+1])]=start
                  new_list.append(start)
                  start+=1
                  if start>=vocab_size:
                      #print(f"in loop exit because start>=vocab_size, start : {start}, vocab_size : {vocab_size}")
                      return enc_map
                else:
                  new_list.append(enc_map[(utf_list[i], utf_list[i+1])])
                i+=2
              else:
                new_list.append(utf_list[i])
                i+=1
            if len(new_list)>1:
                updated_lists.update({tuple(new_list):utf_lists[tuple(utf_list)]})
          
          utf_lists = updated_lists
          pbar.update(1)  
  return enc_map


def train_tokeniser(text, vocab_size, special_tokens=['<|endoftext|>']):
    print("Trainer initiated...")
    spe_enc_map = {special_tokens[i]:i+1+vocab_size for i in range(len(special_tokens))}
    gpt2_pat =  re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    print("GPT pat declared...")
    special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    print("Special pattern declared, starting splitting....")  
    parts = re.split(special_pattern, text)
    print("Splitting completed.....")
    parts = [p for p in parts if p]
    parts = [p for p in parts if p not in special_tokens]
    print("Parts formed....Computing word freq")
    word_map = {}
    for part in tqdm(parts):
        for i in re.findall(gpt2_pat, part):
            if i in word_map:
                word_map[i]+=1
            else:
                word_map.update({i:1})
    utf_lists = {tuple(list(map(int, key.encode("utf-8")))):value for key, value in word_map.items() if len(tuple(list(map(int, key.encode("utf-8")))))>1}
    print("utf_lists formed...")
    print(f"Formed UTF lists of {len(utf_lists)}, Starting Training.....")
    enc_map = bpe(utf_lists, vocab_size)
    spe_enc_map = {special_tokens[i]:i+1+vocab_size for i in range(len(special_tokens))}
    return enc_map, spe_enc_map

def main(args):
   text = Path(args.data_path).read_text(encoding="utf-8")
   enc_map, spe_enc_map = train_tokeniser(text, args.vocab_size)
   out_path_main = f"./trained/iter_{args.version}_main.pkl"
   out_path_spe = f"./trained/iter_{args.version}_spe.pkl"
   with open(out_path_main, "wb") as file:
    pickle.dump(enc_map, file)

   with open(out_path_spe, "wb") as file:
    pickle.dump(spe_enc_map, file)
  
if __name__== "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_path", default="../data/TinyStories-train.txt")
   parser.add_argument("--vocab_size", type=int, default=16384)
   parser.add_argument("--version", type=int, required=True)
   args = parser.parse_args()
   main(args)