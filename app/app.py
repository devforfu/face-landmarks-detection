import argparse
import base64
import io
import importlib.util
from pathlib import Path
import ssl

import cv2 as cv
from jinja2 import Environment, FileSystemLoader
import numpy as np
import PIL.Image
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, JSONResponse
import uvicorn


app = Starlette()
app.mount('/static', StaticFiles(directory='static'))
app.g = {}
templates_dir = str(Path.cwd()/'templates')
env = Environment(loader=FileSystemLoader(templates_dir), trim_blocks=True)


def get_global(name):
    return app.g[name]


def set_global(name, value):
    global app
    app.g[name] = value


@app.route('/')
async def echo(request):
    template = env.get_template('index.html')
    return HTMLResponse(template.render(static_url='/static'))


@app.route('/detect', methods=['POST'])
async def detect(request):
    data = await request.json()
    _, content = data['imgBase64'].split(',')
    decoded = base64.b64decode(content)
    image = read_from_bytes(decoded)
    faces, boxes = detect_faces(image)
    if not faces:
        return JSONResponse({'success': False, 'message': 'no faces detected'})
    model = get_global('model')
    predictions = model.predict(faces)
    points, error = convert_to_absolute(predictions, boxes)
    if error is not None:
        return JSONResponse({'success': False, 'message': error})
    return JSONResponse({'success': True, 'result': points})


def detect_faces(image):
    cascade = get_global('cascade')
    arr = np.asarray(image)
    gray = cv.cvtColor(arr, cv.COLOR_RGBA2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.3, 5)
    faces = [read_from_bytes(gray[y:y+h, x:x+w]) for (x, y, w, h) in boxes]
    return faces, boxes


def convert_to_absolute(predictions, boxes):
    if len(predictions) != len(boxes):
        return None, 'number of predictions is not equal to number of boxes'
    n = len(predictions[0]) // 2
    rescaled = []
    for points, box in zip(predictions, boxes):
        x, y, w, h = box
        points[:n] = x + w*(points[:n] + 1)/2.
        points[n:] = y + h*(points[n:] + 1)/2.
        points = np.round(points).astype(int).tolist()
        box = box.astype(int).tolist()
        rescaled.append({'box': box, 'x': points[:n], 'y': points[n:]})
    return rescaled, None


def read_from_bytes(obj):
    if isinstance(obj, bytes):
        image = PIL.Image.open(io.BytesIO(obj))
    elif isinstance(obj, np.ndarray):
        image = PIL.Image.fromarray(obj)
    else:
        raise TypeError(f'unexpected image type: {type(obj)}')
    return image


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', dest='host', default='0.0.0.0')
    parser.add_argument('--port', dest='port', default=8080, type=int)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--model', dest='model_path', default='code/points_15.py')
    parser.add_argument('--weights', dest='weights_path', default='weights/points_15.pth')
    parser.add_argument('--cert', dest='cert', default=None)
    parser.add_argument('--key', dest='key', default=None)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--cascades-dir', default=None)
    args = parser.parse_args()

    if args.models_dir is None:
        if args.debug:
            args.models_dir = Path.cwd().parent/'models'
        else:
            args.models_dir = Path('/models')

    if args.cascades_dir is None:
        if args.debug:
            args.cascade_dir = Path.cwd().parent/'cascades'
        else:
            args.cascade_dir = Path('/cascades')

    model_path = args.models_dir/args.model_path
    weights_path = args.models_dir/args.weights_path
    model = create_model(model_path, weights_path)
    cascade = cv.CascadeClassifier(str(args.cascade_dir/'haar_face_frontal.xml'))

    app.debug = args.debug
    set_global('model', model)
    set_global('cascade', cascade)

    config = dict(app=app, host=args.host, port=args.port)
    if args.cert is not None and args.key is not None:
        # https://github.com/encode/uvicorn/pull/213
        config.update(dict(
            ssl_version=ssl.PROTOCOL_SSLv23,
            cert_reqs=ssl.CERT_OPTIONAL,
            certfile=args.cert,
            keyfile=args.key
        ))

    uvicorn.run(**config)


def create_model(model_path, weights_path):
    module_name = model_path.stem
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.model_factory(1, 30)
    model.load(weights_path)
    model.train(False)
    return model


if __name__ == '__main__':
    serve()
