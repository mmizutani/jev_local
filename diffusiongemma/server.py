#!/usr/bin/env python3
"""Same-origin Jev API and demo in front of vLLM's structured DiffusionGemma bridge."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import urllib.error
import urllib.request

MAX_BODY = 24 * 1024 * 1024
PAGE = Path(__file__).with_name('demo.html')


def make_handler(bridge_url, bridge_key='', api_key='local-dev', timeout=600):
    bridge_url = bridge_url.rstrip('/')

    class Handler(BaseHTTPRequestHandler):
        server_version = 'JevLocalDiffusionGemma/1.0'

        def send_body(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            self.send_body(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

        def do_GET(self):
            if self.path == '/':
                self.send_body(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/health':
                try:
                    with urllib.request.urlopen(bridge_url + '/health', timeout=3) as response:
                        ok = response.status == 200
                except (OSError, urllib.error.URLError):
                    ok = False
                self.send_json(200 if ok else 503, {'status': 'ok' if ok else 'bridge_unavailable', 'model': 'diffusiongemma'})
            elif self.path == '/v1/models':
                if not self.authorized():
                    return
                self.send_json(200, {'models': [
                    {'name': name, 'description': 'Local structured DiffusionGemma bridge; not TypeSafe Jev weights',
                     'release_date': '2026-09-23'}
                    for name in ('diffusiongemma', 'jev-latest', 'jev-preview')]})
            else:
                self.send_json(404, {'error': {'message': 'Not found'}})

        def authorized(self):
            if hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + api_key):
                return True
            self.send_json(401, {'error': {'type': 'authentication_error', 'message': 'Missing or invalid API key'}})
            return False

        def do_POST(self):
            if self.path not in ('/v1/systemone', '/api/decide'):
                self.send_json(404, {'error': {'message': 'Not found'}})
                return
            if self.path == '/v1/systemone' and not self.authorized():
                return
            if self.path == '/api/decide':
                local_hosts = {f'127.0.0.1:{self.server.server_port}',
                               f'localhost:{self.server.server_port}'}
                host = self.headers.get('Host', '')
                if host not in local_hosts or self.headers.get('Origin') != f'http://{host}':
                    self.send_json(403, {'error': {'type': 'forbidden',
                                                   'message': 'Demo requests must come from the same local origin'}})
                    return
            if self.headers.get_content_type() != 'application/json':
                self.send_json(415, {'error': {'message': 'Content-Type must be application/json'}})
                return
            try:
                length = int(self.headers.get('Content-Length', '-1'))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY or self.headers.get('Transfer-Encoding'):
                self.send_json(413 if length > MAX_BODY else 411, {'error': {'message': 'Expected Content-Length within 24 MiB'}})
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                if body.get('model') not in (None, 'jev-latest', 'jev-preview', 'diffusiongemma'):
                    self.send_json(422, {'error': {'type': 'validation_error', 'message': 'Unknown model'}})
                    return
                body['model'] = 'diffusiongemma'
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            except (ValueError, UnicodeError, RecursionError):
                self.send_json(422, {'error': {'type': 'validation_error', 'message': 'Invalid JSON body'}})
                return
            headers = {'Content-Type': 'application/json'}
            if bridge_key:
                headers['Authorization'] = 'Bearer ' + bridge_key
            request = urllib.request.Request(bridge_url + '/v1/systemone', encoded, headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    self.send_body(response.status, response.read())
            except urllib.error.HTTPError as exc:
                # The structured bridge's validation and model errors are useful to callers.
                try:
                    self.send_body(exc.code, exc.read())
                finally:
                    exc.close()
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                self.send_json(504 if isinstance(exc, (socket.timeout, TimeoutError)) else 502,
                               {'error': {'type': 'server_error', 'message': f'Structured bridge unavailable: {exc}'}})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--bridge-url', default=os.environ.get('JEV_DIFFUSION_BRIDGE_URL', 'http://127.0.0.1:8011'))
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    api_key = os.environ.get('JEV_API_KEY', 'local-dev')
    if not api_key:
        parser.error('JEV_API_KEY must not be empty')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(
        args.bridge_url, os.environ.get('JEV_DIFFUSION_BRIDGE_KEY', ''), api_key, args.timeout))
    print(f'DiffusionGemma API and demo: http://127.0.0.1:{args.port}/\n'
          f'Structured bridge: {args.bridge_url}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
