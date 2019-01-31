import argparse
import ssl
import importlib.util
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
import uvicorn


app = Starlette()
app.mount('/static', StaticFiles(directory='static'))
templates_dir = str(Path.cwd()/'templates')
env = Environment(loader=FileSystemLoader(templates_dir), trim_blocks=True)


@app.route('/')
async def echo(request):
    template = env.get_template('index.html')
    return HTMLResponse(template.render(static_url='/static'))


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', dest='host', default='0.0.0.0')
    parser.add_argument('--port', dest='port', default=8080, type=int)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--model', dest='model_path', default='code/points_15.py')
    parser.add_argument('--weights', dest='weights_path', default='weights/points_15.pth')
    parser.add_argument('--cert', dest='cert', default=None)
    parser.add_argument('--key', dest='key', default=None)
    args = parser.parse_args()

    if args.models_dir is None:
        if args.debug:
            args.models_dir = Path.cwd().parent/'models'
        else:
            args.models_dir = Path('/models')

    model_path = args.models_dir/args.model_path
    weights_path = args.models_dir/args.weights_path
    model = create_model(model_path, weights_path)

    app.debug = args.debug
    app.model = model

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
    model.eval()
    return model


if __name__ == '__main__':
    serve()
