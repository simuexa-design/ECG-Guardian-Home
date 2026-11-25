"""
Root WSGI Entrypoint for Vercel Zero-Config Framework Detection
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app import app


class VercelPathFixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path in ('/api/index', '/api/index.py', '/api', '/api/'):
            orig = (
                environ.get('HTTP_X_FORWARDED_URI') or
                environ.get('HTTP_X_MATCHED_PATH') or
                environ.get('HTTP_X_ORIGINAL_URI') or
                environ.get('RAW_URI') or
                environ.get('REQUEST_URI')
            )
            if orig:
                orig_clean = orig.split('?')[0]
                if orig_clean and orig_clean not in ('/api/index', '/api/index.py', '/api'):
                    environ['PATH_INFO'] = orig_clean
                else:
                    environ['PATH_INFO'] = '/'
            else:
                environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)


app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

application = app
handler = app
