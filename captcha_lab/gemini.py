"""Small REST client. Secrets never appear in repr, URLs, saved results or exceptions."""
import base64
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from .data import ROOT, png_bytes


class GeminiError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.safe_message = message


@dataclass(repr=False)
class GeminiClient:
    api_key: str = field(repr=False)
    proxy_url: str = field(default='', repr=False)
    model: str = 'gemini-flash-latest'

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / '.env.local', override=False)
        key = os.getenv('GEMINI_API_KEY', '').strip()
        if not key:
            raise GeminiError('GEMINI_API_KEY is missing in local configuration')
        proxy = os.getenv('GEMINI_PROXY_URL', '').strip()
        if proxy:
            parsed = urlsplit(proxy)
            if parsed.scheme not in {'socks5','socks5h','http','https'} or not parsed.hostname or not parsed.port:
                raise GeminiError('Invalid proxy configuration')
            # DNS is resolved by the SOCKS proxy, with no direct fallback.
            if proxy.startswith('socks5://'):
                proxy = 'socks5h://' + proxy[len('socks5://'):]
        model = os.getenv('GEMINI_MODEL', 'gemini-flash-latest').removeprefix('models/')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', model):
            raise GeminiError('Invalid model identifier')
        return cls(key, proxy, model)

    def __repr__(self):
        return f'GeminiClient(model={self.model!r}, proxy_enabled={bool(self.proxy_url)})'

    def _request(self, method, endpoint, payload=None):
        try:
            with requests.Session() as session:
                session.trust_env = False
                if self.proxy_url:
                    session.proxies.update(http=self.proxy_url, https=self.proxy_url)
                response = session.request(method, 'https://generativelanguage.googleapis.com/v1beta/' + endpoint,
                                           headers={'X-goog-api-key': self.api_key}, json=payload,
                                           timeout=(15, 90), allow_redirects=False)
                if response.status_code != 200:
                    try:
                        code = response.json().get('error', {}).get('status', 'UNKNOWN')
                    except ValueError:
                        code = 'NON_JSON'
                    # Never include response body or request/proxy details.
                    code = code if re.fullmatch(r'[A-Z_]+', str(code)) else 'UNKNOWN'
                    raise GeminiError(f'Gemini HTTP {response.status_code}: {code}')
                return response.json()
        except GeminiError:
            raise
        except requests.RequestException:
            raise GeminiError('Network/proxy connection failed; no direct fallback attempted') from None

    def list_models(self):
        result, page = [], ''
        for _ in range(20):
            data = self._request('GET', 'models?pageSize=100' + ('&pageToken=' + requests.utils.quote(page, safe='') if page else ''))
            result.extend(x['name'].removeprefix('models/') for x in data.get('models', [])
                          if 'generateContent' in x.get('supportedGenerationMethods', []))
            page = data.get('nextPageToken', '')
            if not page:
                return result
        raise GeminiError('Unexpected model-list pagination limit')

    def generate(self, text, image=None):
        parts = [{'text': text}]
        if image is not None:
            parts.append({'inlineData': {'mimeType': 'image/png',
                                        'data': base64.b64encode(png_bytes(image)).decode('ascii')}})
        payload = {'contents': [{'parts': parts}],
                   'generationConfig': {'temperature': 0, 'maxOutputTokens': 256}}
        start = time.perf_counter()
        data = self._request('POST', f'models/{self.model}:generateContent', payload)
        candidates = data.get('candidates', [])
        if not candidates:
            raise GeminiError('No candidate returned; inspect account/model policy outside this notebook')
        candidate = candidates[0]
        if candidate.get('finishReason') not in {None, 'STOP'}:
            reason = candidate.get('finishReason')
            reason = reason if re.fullmatch(r'[A-Z_]+', str(reason)) else 'UNKNOWN'
            raise GeminiError('Response did not finish normally: ' + reason)
        answer = ''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought'))
        if not answer.strip():
            raise GeminiError('Empty response')
        return dict(text=answer.strip(), elapsed_ms=(time.perf_counter()-start)*1000,
                    model_version=data.get('modelVersion', self.model), usage=data.get('usageMetadata',{}))

    def recognize(self, image):
        result = self.generate('Transcribe only the characters in this image, in order. '
                               'Return just the string, preserving case and leading zeros. '
                               'If unreadable, return exactly UNREADABLE.', image)
        answer = result['text']
        if answer == 'UNREADABLE':
            raise GeminiError('Model abstained')
        if not re.fullmatch(r'[A-Za-z0-9]{1,32}', answer):
            raise GeminiError('Response is not a plain alphanumeric transcription')
        return answer, None
