import regex as re

def encoder(utf_lists, enc_map):
  updated_lists = []
  count=0
  for utf_list in utf_lists:
    new_list = []
    if len(utf_list)<=1:
      updated_lists.append(utf_list)
      continue
    maps = [enc_map.get((utf_list[i], utf_list[i+1])) for i in range(len(utf_list)-1) if enc_map.get((utf_list[i], utf_list[i+1]))]
    if len(maps)==0:
      updated_lists.append(utf_list)
      continue
    mini = min(maps)
    i=0
    while i<len(utf_list):
      if i==len(utf_list)-1:
        new_list.append(utf_list[i])
        break
      if enc_map.get((utf_list[i], utf_list[i+1]))==mini:
        new_list.append(mini)
        i+=2
      else:
        new_list.append(utf_list[i])
        i+=1
    updated_lists.append(new_list)
    count+=1

  if count==0:
    return updated_lists

  return encoder(updated_lists, enc_map)

def decoder(encoded_list, dec_map):
  checker = [i for i in encoded_list if i in dec_map]
  if len(checker)==0:
    return encoded_list
  encoded_lis = encoded_list.copy()
  decoded_list = []
  for el in encoded_lis:
    if el in dec_map:
      decoded_list+=list(dec_map[el])
    else:
      decoded_list.append(el)
  return decoder(decoded_list, dec_map)

def encode_text(text, enc_map, spe_enc_map, special_tokens=['<|endoftext|>']):
  # for tok in special_tokens:
  #   text = text.strip(tok)
  special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
  parts = re.split(special_pattern, text)
  parts = [p for p in parts if p]
  gpt2_pat =  re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
  split_text = []
  for part in parts:
    if part not in special_tokens:
      split_text.extend(re.findall(gpt2_pat, part))
    else:
      split_text.append(part)

  tokens =[]
  for t in split_text:
    if t not in special_tokens:
      utf_lists = [list(map(int, t.encode("utf-8")))]
      encoded_lists = encoder(utf_lists, enc_map)
      for l in encoded_lists:
        tokens+=l
    else:
      tokens.append(spe_enc_map[t])

  return tokens

def decode_text(encoded, encf_map, spe_map):
  dec_map = {value:key for key,value in encf_map.items()}
  dec_spe_map = {value:key for key,value in spe_map.items()}
  dec_utf = decoder(encoded, dec_map)
  dec_utf = [i for i in dec_utf if i<256]
  # for ut in dec_utf:
  #   if ut>=256:
  #     dec_text+=dec_spe_map.get(ut)
  #   else:
  #     dec_text += bytes([ut]).decode("utf-8", errors="ignore")
  dec_text = bytes(dec_utf).decode("utf-8", errors="ignore")
  return dec_text
