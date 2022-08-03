import subprocess
from flask import Flask, flash, request, redirect, url_for, session
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import logging
import csv

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

app.debug = True

UPLOAD_FOLDER = 'tmp/'  
OUTPUT_FOLDER = 'out/'
# ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f_dir = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        o_dir = os.path.join(app.config['OUTPUT_FOLDER'],'output.txt')
        f.save(f_dir)

        return predict(f_dir,o_dir)

# # file_dir = 'tmp/dev.txt'
def predict(file_dir, output_dir):
    arg = ["python", "predict.py", "--model_dir", "./models/original_train_dev/phobert_em_es_base_maxlen_512_epochs_100", "--eval_data_file", file_dir, "--save_output", output_dir, "--id2label", "data/id2label.txt", "--model_type", "es"]
    output = subprocess.Popen(arg)
    return 'ok'
    
# def convert_ner_to_re(input_path, tmp_out):
#     csv_reader = csv.reader(open(input_path), delimiter='\t')
#     for row in csv_reader:
#         tmp = []
#         text = row[1]
#         labels = row[3].split(',')
#         positions = list(eval(row[2]))
#         for position,label in zip(positions, labels):
#             # print(position, label)
#             tmp.append([position, label])
#         with open(tmp_out, 'a+', encoding='utf-8') as re_in:
#             for i in range(len(tmp)-1):
#                 for j in range(i+1, len(tmp)):
#                     text_out = str(tmp[i][0][0]) + '\t' + str(tmp[i][0][1])+ '\t' + str(tmp[j][0][0])+ '\t' + str(tmp[j][0][1])+ '\t' + tmp[i][1]+ '\t' + tmp[j][1]+ '\t' + text + '\n'
#                     # print(text_out)
#                     re_in.write(text_out)
#     return 0
# convert_ner_to_re('test_convert/ner.csv', 'test_convert/out_re.txt')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)

    