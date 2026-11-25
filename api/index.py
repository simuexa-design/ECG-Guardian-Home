"""
Vercel Serverless Function Entrypoint
Routes all incoming web traffic to the unified ECG Guardian Flask Application.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app import app


class VercelPathFixMiddleware:
    """
    WSGI Middleware to fix PATH_INFO on Vercel Serverless Functions.
    Vercel internal rewrites (/api/index or /api/index.py) can mask the
    client's intended route (e.g. /, /api/auth/login, /api/status, /?view=...).
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')

        # When Vercel sets the rewritten internal function path as PATH_INFO:
        if path in ('/api/index', '/api/index.py', '/api', '/api/'):
            orig = (
                environ.get('HTTP_X_FORWARDED_URI') or
                environ.get('HTTP_X_ORIGINAL_URI') or
                environ.get('HTTP_X_MATCHED_PATH') or
                environ.get('RAW_URI') or
                environ.get('REQUEST_URI')
            )
            if orig:
                orig_clean = orig.split('?')[0]
                if orig_clean and orig_clean not in ('/api/index', '/api/index.py', '/api', '/api/'):
                    environ['PATH_INFO'] = orig_clean
                else:
                    environ['PATH_INFO'] = '/'
            else:
                environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)


# Apply middleware to Flask WSGI callable
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

# Export application across all standard serverless conventions
application = app
handler = app
