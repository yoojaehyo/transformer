from sacremoses import MosesTokenizer
import json

def tokenize(sentences, lang="en") :
	tok = MosesTokenizer(lang)
	
	result = []
	for sentence in sentences :
		result.append(tok.tokenize(sentence))
		
	return result

def build_vocab(sentences, min_count=2) :
	vocab = [("<unk>", 0)]
	
	for sentence in sentences :
		for word in sentence :
			is_new = True
			
			for idx, (w, cnt) in enumerate(vocab) :
				if word == w :
					is_new = False
					vocab[idx] = (w, cnt+1)
					break
					
			if is_new :
				vocab.append((word, 1))
	
	# except "<unk>"
	for i in range(len(vocab)-1, 0, -1) :
		_, cnt = vocab[i]
		if cnt < min_count :
			del vocab[i]
			vocab[0] = ("<unk>", vocab[0][1] + 1)
	
	return [w for w, _ in vocab]


for filename in ["train.de", "train.en", "val.de", "val.en", "test.de", "test.en"] :
	
	with open("./data/" + filename) as f :
		new_example = tokenize(f.readlines())

		with open('./data/' + filename + '.json', 'w', encoding='utf-8') as make_file:
			json.dump(new_example, make_file, indent="\t")

		if filename.split(".")[0] == "train" :
			vocab = build_vocab(new_example)
			with open('./data/vocab.' + filename.split(".")[1] + '.json', 'w', encoding='utf-8') as make_file:
				json.dump(vocab, make_file, indent="\t")
