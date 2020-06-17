import requests
import csv
import logging

'''
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

data = {'title': '极米被诉侵权产品 Z6 系列型号投影仪',
        'content': "酒鬼酒称，本公司严禁在产品中添加甜蜜素，本公司已经提请相关市场监管部门对本公司市场流通产品进行全面检测，并第一时间向社会公布检测结果。\n所采用的技术方案构成侵权。"}
r = requests.post("http://localhost:8001/predict", data=data)
print(r.json())
