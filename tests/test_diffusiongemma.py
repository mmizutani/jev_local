"""HTTP contract tests against a local fake bridge; no model or GPU required."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from diffusiongemma.server import make_handler


class Bridge(BaseHTTPRequestHandler):
    requests = []
    status = 200

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.requests.append((self.path, self.headers.get('Authorization'), json.loads(body)))
        response = ({'model': 'diffusiongemma', 'answers': {
            'yes': {'type': 'noul', 'noul': .82},
            'route': {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': .7, 'sales': .3}, 'confidence': .12},
            'risk': {'type': 'score', 'score': 1.4, 'legend': {'0': 'low', '1': 'medium', '2': 'high'},
                     'probabilities': {'0': .1, '1': .4, '2': .5}, 'confidence': .05}},
            'usage': {'input_tokens': 33, 'output_tokens': 7},
            'diagnostics': {'timing': {'total_ms': 123, 'reads': 1}}}
            if self.status == 200 else {'error': {'type': 'validation_error', 'message': 'bad question'}})
        data = json.dumps(response).encode()
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ThreadingHTTPServer(('127.0.0.1', 0), Bridge)
        cls.proxy = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
            f'http://127.0.0.1:{cls.bridge.server_port}', 'bridge-secret', 'local-dev'))
        cls.threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (cls.bridge, cls.proxy)]
        for thread in cls.threads:
            thread.start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.proxy, cls.bridge):
            server.shutdown()
            server.server_close()

    def setUp(self):
        Bridge.requests.clear()
        Bridge.status = 200
        self.payload = {'model': 'jev-latest', 'state': {'message': 'refund please'}, 'samples': 1, 'steps': 1,
                        'questions': {'yes': {'type': 'noul', 'instructions': 'Refund?'},
                                      'route': {'type': 'choice', 'instructions': 'Team?', 'criteria': {'billing': 'Refund', 'sales': None}},
                                      'risk': {'type': 'score', 'instructions': 'Risk?', 'criteria': ['low', 'medium', 'high']}}}

    def request(self, path, payload, key=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        if origin:
            headers['Origin'] = origin
        return urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{self.proxy.server_port}{path}',
            json.dumps(payload).encode(), headers), timeout=5)

    def test_forwards_jev_schema_and_preserves_distributions_diagnostics(self):
        with self.request('/v1/systemone', self.payload, 'local-dev') as response:
            result = json.load(response)
        self.assertEqual(Bridge.requests, [('/v1/systemone', 'Bearer bridge-secret',
                                            dict(self.payload, model='diffusiongemma'))])
        self.assertEqual(result['answers']['yes']['noul'], .82)
        self.assertEqual(result['answers']['route']['probabilities']['billing'], .7)
        self.assertEqual(result['answers']['risk']['legend']['2'], 'high')
        self.assertEqual(result['diagnostics']['timing']['total_ms'], 123)

    def test_demo_uses_same_origin_without_browser_secret(self):
        with self.request('/api/decide', self.payload,
                          origin=f'http://127.0.0.1:{self.proxy.server_port}') as response:
            self.assertEqual(json.load(response)['model'], 'diffusiongemma')
        with urllib.request.urlopen(f'http://127.0.0.1:{self.proxy.server_port}/') as response:
            page = response.read().decode()
        self.assertIn('DiffusionGemma', page)
        self.assertNotIn('bridge-secret', page)

    def test_upstream_validation_error_is_visible(self):
        Bridge.status = 422
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/decide', self.payload,
                         origin=f'http://127.0.0.1:{self.proxy.server_port}')
        self.assertEqual(error.exception.code, 422)
        self.assertEqual(json.load(error.exception)['error']['message'], 'bad question')
        error.exception.close()

    def test_demo_rejects_foreign_or_missing_origin_before_bridge(self):
        for origin in ('https://evil.example', None):
            with self.subTest(origin=origin):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request('/api/decide', self.payload, origin=origin)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
        rebinding = urllib.request.Request(
            f'http://127.0.0.1:{self.proxy.server_port}/api/decide',
            json.dumps(self.payload).encode(),
            {'Content-Type': 'application/json', 'Host': f'evil.example:{self.proxy.server_port}',
             'Origin': f'http://evil.example:{self.proxy.server_port}'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(rebinding)
        self.assertEqual(error.exception.code, 403)
        error.exception.close()
        self.assertFalse(Bridge.requests)

    def test_api_requires_local_key(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/v1/systemone', self.payload)
        self.assertEqual(error.exception.code, 401)
        error.exception.close()
        self.assertFalse(Bridge.requests)

    def test_models_endpoint_lists_accepted_names(self):
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.proxy.server_port}/v1/models',
            headers={'Authorization': 'Bearer local-dev'})
        with urllib.request.urlopen(request) as response:
            names = [model['name'] for model in json.load(response)['models']]
        self.assertEqual(names, ['diffusiongemma', 'jev-latest', 'jev-preview'])


if __name__ == '__main__':
    unittest.main()
