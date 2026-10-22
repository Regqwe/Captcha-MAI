"""Screen interpretation and bounded watcher; no Android imports or paid services."""
from dataclasses import MISSING, dataclass, fields
from collections import deque
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Profile:
    package: str | None
    title: str
    footer: str
    submit: str
    refresh: str
    input: str
    code_pattern: str
    roi: tuple
    aspect_range: tuple
    min_score: float
    stable_frames: int
    max_attempts: int
    poll_seconds: float
    response_seconds: float
    success: str | None = None

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        allowed = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
        if set(data) - allowed or not required <= set(data):
            raise ValueError('Profile fields do not match phone-profile.json')
        p = cls(**data)
        for name in ('title', 'footer', 'submit', 'refresh', 'input', 'code_pattern'):
            re.compile(getattr(p, name), re.I)
        if p.success:
            re.compile(p.success, re.I)
        x1, y1, x2, y2 = p.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('ROI must be normalized x1,y1,x2,y2')
        if not (0 < p.aspect_range[0] < p.aspect_range[1]):
            raise ValueError('Invalid screen aspect range')
        if not (0 <= p.min_score <= 1 and p.stable_frames >= 2 and p.max_attempts >= 1
                and p.poll_seconds >= .2 and p.response_seconds >= p.poll_seconds):
            raise ValueError('Invalid watcher limits')
        return p

    def accepts(self, text, score):
        # Never delete letters/noise, invent substitutions, or drop leading zeros.
        return bool(re.fullmatch(self.code_pattern, text) and score is not None
                    and math.isfinite(score) and score >= self.min_score)


@dataclass(frozen=True)
class Element:
    bounds: tuple
    resource_id: str
    class_name: str
    text: str


@dataclass
class Screen:
    image: Image.Image
    hierarchy: str
    package: str


@dataclass
class View:
    state: str
    crop: Image.Image | None = None
    field: Element | None = None
    button: Element | None = None
    refresh: Element | None = None


def _bounds(node, size):
    match = re.fullmatch(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', node.get('bounds', ''))
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    w, h = size
    return (x1, y1, x2, y2) if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h else None


def inspect(screen, profile):
    """Require title + footer + compatible layout before using a configured ROI."""
    if profile.package and screen.package != profile.package:
        return View('other_app')
    root = ET.fromstring(screen.hierarchy)
    if root.tag != 'hierarchy' or not list(root.iter('node')):
        raise ValueError('Empty or invalid Android hierarchy')
    parents = {child: node for node in root.iter() for child in node}
    nodes = [n for n in root.iter('node')
             if n.get('visible-to-user', 'true') == 'true' and _bounds(n, screen.image.size)]

    def text(n):
        return ' '.join(n.get(k, '') for k in ('text', 'content-desc', 'hint')).strip()

    def matches(pattern):
        return [n for n in nodes if re.search(pattern, text(n), re.I)]

    titles = matches(profile.title)
    if not titles:
        return View('absent')
    if len(titles) == 1 and profile.success and len(matches(profile.success)) == 1:
        return View('accepted')
    footers = matches(profile.footer)
    if len(titles) != 1 or len(footers) != 1:
        return View('unsupported_ui')
    w, h = screen.image.size
    if not profile.aspect_range[0] <= h / w <= profile.aspect_range[1]:
        return View('unsupported_layout')
    box = tuple(int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(profile.roi))
    top = _bounds(titles[0], (w, h))[3]
    bottom = _bounds(footers[0], (w, h))[1]
    if not (top <= box[1] < box[3] <= bottom):
        return View('roi_outside_anchors')
    crop = screen.image.crop(box).convert('RGB')

    edits = [n for n in nodes if n.get('class') == 'android.widget.EditText'
             and n.get('enabled', 'true') == 'true' and _bounds(n, (w, h))[1] >= bottom]
    hinted = [n for n in edits if re.search(profile.input, text(n), re.I)]
    candidates = hinted or edits

    def element(n):
        return Element(_bounds(n, (w, h)), n.get('resource-id', ''), n.get('class', ''), n.get('text', ''))

    def button(pattern):
        buttons = {}
        for n in nodes:
            if not re.fullmatch(pattern, n.get('text', '') or n.get('content-desc', ''), re.I):
                continue
            while n is not root and n.get('clickable') != 'true':
                n = parents[n]
            if n is not root and n.get('enabled', 'true') == 'true' and _bounds(n, (w, h)):
                el = element(n)
                if el.bounds[1] >= bottom:
                    buttons[el.bounds] = el
        return next(iter(buttons.values())) if len(buttons) == 1 else None
    field = element(candidates[0]) if len(candidates) == 1 else None
    # A disabled Next button is normal before typing. Input is enough for readiness.
    return View('ready' if field else 'input_missing', crop, field, button(profile.submit), button(profile.refresh))


def signature(image):
    return np.asarray(image.convert('L').resize((64, 24)), dtype=np.float32)


def same_image(a, b):
    return a is not None and b is not None and float(np.mean(np.abs(a - b))) <= 1.5


def fingerprint(image):
    return hashlib.sha256((signature(image) < 190).tobytes()).hexdigest()[:16]


class Watcher:
    """One submission per stable image, bounded attempts per uninterrupted challenge."""
    def __init__(self, device, predictor, profile, *, observe=False, emit=None, clock=time.monotonic):
        self.device, self.predict, self.profile = device, predictor, profile
        self.observe, self.emit, self.clock = observe, emit or (lambda **kw: None), clock
        self.previous = None
        self.stable = self.attempts = self.absent = 0
        self.handled = deque(maxlen=64)
        self.pending = None
        self.last_status = None
        self.package = None

    def event(self, status, **data):
        if data or status != self.last_status:
            self.emit(status=status, **data)
        self.last_status = status
        return status

    def tick(self):
        screen = self.device.capture()
        self.package = screen.package
        view = inspect(screen, self.profile)
        now = self.clock()
        if view.state == 'accepted':
            if self.pending:
                self.pending = None
                return self.event('accepted')
            return self.event('already_accepted')
        if view.state == 'absent':
            self.absent += 1
            self.previous, self.stable = None, 0
            if self.pending and self.absent >= 3:
                self.pending = None
                self.event('screen_left_unverified')
            if self.absent >= 3:
                self.attempts = 0
                self.handled.clear()
            return self.event('waiting')
        # Empty XML, other app, changed layout and missing controls never signal success.
        self.absent = 0
        if view.state != 'ready':
            self.previous, self.stable = None, 0
            return self.event(view.state)
        current = signature(view.crop)
        self.stable = self.stable + 1 if same_image(current, self.previous) else 1
        self.previous = current
        if self.pending:
            sent_at, _ = self.pending
            if now - sent_at < self.profile.response_seconds:
                return self.event('awaiting_response')
            self.pending = None
            self.event('response_not_confirmed')
        if self.stable < self.profile.stable_frames:
            return self.event('stabilizing')
        if any(same_image(current, old) for old in self.handled):
            return self.event('already_seen')
        if self.attempts >= self.profile.max_attempts:
            return self.event('attempt_limit')
        start = self.clock()
        raw, score = self.predict(view.crop)
        if not self.profile.accepts(raw, score):
            self.handled.append(current)
            if not self.observe and view.refresh:
                self.attempts += 1
                if self.device.refresh(view, self.profile):
                    self.pending = (self.clock(), current)
                    return self.event('refreshed', raw=raw, score=score, attempt=self.attempts)
            return self.event('rejected_prediction', raw=raw, score=score)
        if self.observe:
            self.handled.append(current)
            return self.event('observed', raw=raw, score=score, image_id=fingerprint(view.crop))
        # Re-read both screenshot and UI after inference; never type a stale answer.
        fresh = inspect(self.device.capture(), self.profile)
        if fresh.state != 'ready' or not same_image(current, signature(fresh.crop)):
            self.previous, self.stable = None, 0
            return self.event('screen_changed')
        self.attempts += 1
        self.handled.append(current)
        if not self.device.submit(fresh, raw, self.profile):
            # Adapter returns False only before clicking; allow another bounded try.
            # Exceptions retain the handled image because the click outcome is uncertain.
            self.handled.pop()
            self.previous, self.stable = None, 0
            return self.event('input_not_submitted', attempt=self.attempts)
        self.pending = (self.clock(), current)
        return self.event('submitted', raw=raw, score=score, attempt=self.attempts,
                          image_id=fingerprint(view.crop), inference_and_action_ms=(self.clock()-start)*1000)
