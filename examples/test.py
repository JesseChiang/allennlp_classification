import requests
import csv
import logging

logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

predict = open("C:\\Users\\jesse\\Documents\\Datago\\data\\tmp\\I6_predict.csv", 'w')
writer = csv.DictWriter(predict, fieldnames=['text', 'label', 'prob'], dialect='excel')
writer.writeheader()

with open("C:\\Users\\jesse\\Documents\\Datago\\data\\I6.csv", 'r') as f:
    reader = csv.DictReader(f)
    logging.info(f'读取')
    for row in reader:
        logging.info(f'读取{row["text"]}')
        data = {'sentence': row['text']}
        logging.info(f'发送请求')
        r = requests.post("http://localhost:8001/predict", data=data)
        logging.info(f'写入{row["text"]}')
        writer.writerow({'text': row['text'],
                         'label': r.json()['label'],
                         'prob': r.json()['prob']})

predict.close()
'''
data = {'sentence': "极米被诉侵权产品 Z6 系列型号投影仪包括Z6和Z6X被诉侵权产品所采用的技术方案构成侵权。"}
r = requests.post("http://localhost:8001/predict", data=data)
print(r.json()['label'])
'''