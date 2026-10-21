import math

import base64
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from decoder import decode_redis_tape
from model import LogisticModel
from store import Store

store = Store()
model = None

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global model
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length))
        
        if self.path == '/fit':
            db_path = data['db_path']
            epochs = data['epochs']
            lr = data['lr']
            
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            for row in c.execute("SELECT blob FROM records"):
                lbl, feats = decode_redis_tape(row[0])
                store.add_record(lbl, feats)
            conn.close()
            
            records = store.get_records()
            if not records:
                self.send_response(400)
                self.end_headers()
                return
                
            if model is None or len(model.weights) != len(records[0][1]):
                model = LogisticModel(len(records[0][1]))
                
            losses = []
            for _ in range(epochs):
                loss_sum = 0
                for lbl, feats in records:
                    model.train_step(feats, lbl, lr)
                    pred = model.predict(feats)
                    pred = max(min(pred, 0.999), 0.001)
                    loss_sum += - (lbl * math.log(pred) + (1 - lbl) * math.log(1 - pred))
                losses.append(loss_sum / len(records))
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'losses': losses}).encode())
            
        elif self.path == '/checkpoint':
            key = data['key']
            store.save_checkpoint(key, model.weights, model.bias)
            self.send_response(200)
            self.end_headers()
            
        elif self.path == '/resume':
            key = data['key']
            res = store.load_checkpoint(key)
            if res:
                model.weights, model.bias = res
            self.send_response(200)
            self.end_headers()
            
        elif self.path == '/score':
            blob = base64.b64decode(data['blob'])
            _, feats = decode_redis_tape(blob)
            score = model.predict(feats)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'score': score}).encode())
            
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('', 8080), Handler)
    server.serve_forever()
