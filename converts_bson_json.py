import json
import bson
from bson import json_util

i = r"data/geo_false"

INPUTF = i + ".bson"
OUTPUTF = i + ".json"

input_file = open(INPUTF, 'rb')
output_file = open(OUTPUTF, 'w+')

raw = (input_file.read())
datas = bson.decode_all(raw)
# print(datas)
# json.dump(datas, output_file, default=json_util.default)
json.dump(datas, output_file, default=str)

