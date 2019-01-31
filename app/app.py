import argparse
import os
from os.path import join

from jinja2 import Environment, FileSystemLoader
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse, HTMLResponse
import uvicorn


app = Starlette()
app.mount('/static', StaticFiles(directory='static'))
root_dir = join(os.getcwd(), 'templates')
env = Environment(loader=FileSystemLoader(root_dir), trim_blocks=True)


@app.route('/')
def echo(request):
    template = env.get_template('index.html')
    return HTMLResponse(template.render(static_url='/static'))


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument('-host', dest='host', default='0.0.0.0')
    parser.add_argument('-port', dest='port', default=8080, type=int)
    parser.add_argument('-debug', action='store_true', default=False)
    args = parser.parse_args()
    app.debug = args.debug
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    serve()
