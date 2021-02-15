import json


with open('files.json') as f:
    data = json.load(f)


with open('models.tsv','w') as f:
	f.write('cell_type\tmodel link\tmd5\n')
	for file in data['files']:
		if '.pt' in file['links']['self']:
			cell_type = file['links']['self'].split('/')[-1].split('_')[0]
			f.write(cell_type+'\t')
			f.write(file['links']['self']+'?download=1'+'\t')
			f.write(file['checksum'].split('md5:')[1]+'\n')
