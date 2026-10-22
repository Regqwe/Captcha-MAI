"""State/transport contracts with synthetic UI XML, not device or accuracy evidence."""
from dataclasses import replace
import unittest

from PIL import Image, ImageOps

from captcha_lab.android import Android, choose_serial
from captcha_lab.data import ROOT
from captcha_lab.phone import Profile, Screen, Watcher, inspect


PROFILE = Profile.load(ROOT/'phone-profile.json')


def screen(image=None, *, title=True, enabled=True, field=True, code='', package='test.app'):
    # Geometry approximates the real supplied screenshot; this XML is synthetic.
    xml = '<hierarchy><node class="android.widget.FrameLayout" bounds="[0,0][1080,2220]">'
    if title:
        xml += '<node text="Confirm you\'re human" bounds="[40,290][1000,380]"/>'
    xml += '<node text="Can\'t read this text?" bounds="[40,810][1000,880]"/>'
    if field:
        xml += (f'<node class="android.widget.EditText" text="{code}" hint="Enter the code" '
                'resource-id="test:id/code" bounds="[80,970][1000,1120]"/>')
    xml += (f'<node text="Next" clickable="true" enabled="{str(enabled).lower()}" '
            'bounds="[40,1900][1040,2040]"/></node></hierarchy>')
    return Screen(image if image is not None else Image.open(ROOT/'datasets/phone/screen.png').convert('RGB'),
                  xml, package)


class Device:
    def __init__(self):
        self.frame = screen()
        self.submissions = []

    def capture(self):
        return self.frame

    def submit(self, view, code, profile):
        self.submissions.append(code)
        return True


class PhoneContracts(unittest.TestCase):
    def test_crop_keeps_all_six_digits_and_full_right_edge(self):
        view = inspect(screen(), PROFILE)
        self.assertEqual(view.state, 'ready')
        self.assertEqual(view.crop.size, (864, 267))
        self.assertEqual(view.field.resource_id, 'test:id/code')

    def test_format_and_confidence_do_not_modify_text(self):
        self.assertTrue(PROFILE.accepts('001234', .95))
        for raw, score in [('W9H5K', .99), ('12345x6', .99), ('123456', .2), ('123456', float('nan'))]:
            self.assertFalse(PROFILE.accepts(raw, score))
        letters = replace(PROFILE, code_pattern='[A-Z0-9]{5}')
        self.assertTrue(letters.accepts('W9H5K', .99))

    def test_missing_field_and_disabled_button_have_no_coordinate_fallback(self):
        self.assertEqual(inspect(screen(field=False), PROFILE).state, 'input_missing')
        view = inspect(screen(enabled=False), PROFILE)
        self.assertEqual(view.state, 'ready')
        self.assertIsNone(view.button)

    def test_wrong_app_layout_or_empty_xml_cannot_be_ready(self):
        self.assertEqual(inspect(screen(), replace(PROFILE, package='another.app')).state, 'other_app')
        portrait = screen()
        portrait.image = portrait.image.rotate(90, expand=True)
        self.assertNotEqual(inspect(portrait, PROFILE).state, 'ready')
        with self.assertRaises(ValueError):
            inspect(Screen(Image.new('RGB', (1080,2220)), '<hierarchy/>', 'test.app'), PROFILE)

    def test_stability_deduplication_and_no_false_success(self):
        d = Device()
        events = []
        now = [0.]
        w = Watcher(d, lambda im: ('001234', .99), PROFILE, emit=lambda **e: events.append(e), clock=lambda: now[0])
        self.assertEqual(w.tick(), 'stabilizing')
        self.assertEqual(w.tick(), 'submitted')
        self.assertEqual(d.submissions, ['001234'])
        now[0] = 20
        self.assertEqual(w.tick(), 'already_seen')
        d.frame = screen(title=False)
        for _ in range(3):w.tick()
        self.assertTrue(any(e['status'] == 'response_not_confirmed' for e in events))
        self.assertFalse(any(e['status'] == 'success' for e in events))
        self.assertEqual(len(d.submissions), 1)

    def test_changed_image_during_inference_is_never_submitted(self):
        d = Device()
        def predict(im):
            d.frame = screen(ImageOps.invert(d.frame.image))
            return '123456', .99
        w = Watcher(d, predict, PROFILE)
        w.tick()
        self.assertEqual(w.tick(), 'screen_changed')
        self.assertEqual(d.submissions, [])

    def test_observe_is_read_only_and_invalid_predictions_are_rejected(self):
        for raw, mode, status in [('001234', True, 'observed'), ('W9H5K', False, 'rejected_prediction')]:
            d = Device()
            w = Watcher(d, lambda im: (raw, .99), PROFILE, observe=mode)
            w.tick()
            self.assertEqual(w.tick(), status)
            self.assertEqual(d.submissions, [])

    def test_attempt_limit_does_not_reset_on_a_new_image(self):
        d = Device()
        now = [0.]
        w = Watcher(d, lambda im: ('123456', .99), replace(PROFILE, max_attempts=1), clock=lambda: now[0])
        w.tick();w.tick()
        d.frame = screen(ImageOps.invert(d.frame.image))
        now[0] = 20
        w.tick()
        self.assertEqual(w.tick(), 'attempt_limit')
        self.assertEqual(len(d.submissions), 1)

    def test_refresh_requires_a_named_clickable_control_and_uses_attempt_budget(self):
        d = Device()
        s = screen()
        s.hierarchy = s.hierarchy.replace('</node></hierarchy>',
            '<node text="Get a new code" clickable="true" bounds="[40,890][400,940]"/></node></hierarchy>')
        d.frame = s
        refreshed = []
        d.refresh = lambda view, profile:refreshed.append(view.refresh.bounds) or True
        w = Watcher(d, lambda im: ('bad', .99), PROFILE)
        w.tick()
        self.assertEqual(w.tick(), 'refreshed')
        self.assertEqual(refreshed, [(40,890,400,940)])
        self.assertEqual(w.attempts, 1)

    def test_disappearance_is_unverified_even_after_submission(self):
        d = Device()
        events = []
        w = Watcher(d, lambda im: ('001234', .99), PROFILE, emit=lambda **e:events.append(e))
        w.tick();w.tick()
        d.frame = screen(title=False)
        for _ in range(3):w.tick()
        self.assertIn('screen_left_unverified', [e['status'] for e in events])
        self.assertEqual(w.attempts, 0)

    def test_explicit_success_marker_is_required_for_accepted_status(self):
        d = Device()
        profile = replace(PROFILE, success='^Captcha is passed successfully!$')
        w = Watcher(d, lambda im: ('001234', .99), profile)
        w.tick();w.tick()
        d.frame.hierarchy = d.frame.hierarchy.replace('</node></hierarchy>',
            '<node text="Captcha is passed successfully!" bounds="[80,1150][1000,1220]"/></node></hierarchy>')
        self.assertEqual(w.tick(), 'accepted')
        self.assertEqual(w.tick(), 'already_accepted')
        # A fresh process may see the marker, but cannot attribute it to its own submission.
        fresh = Watcher(d, lambda im: ('001234', .99), profile)
        self.assertEqual(fresh.tick(), 'already_accepted')

    def test_known_no_click_failure_can_retry_but_uncertain_exception_cannot(self):
        d = Device()
        d.submit = lambda *args:False
        w = Watcher(d, lambda im: ('123456', .99), replace(PROFILE, max_attempts=1))
        w.tick()
        self.assertEqual(w.tick(), 'input_not_submitted')
        w.tick()
        self.assertEqual(w.tick(), 'attempt_limit')
        d = Device()
        def uncertain(*args):raise RuntimeError('uncertain transport outcome')
        d.submit = uncertain
        w = Watcher(d, lambda im: ('123456', .99), PROFILE)
        w.tick()
        with self.assertRaises(RuntimeError):w.tick()
        self.assertEqual(w.tick(), 'already_seen')

    def test_device_selection_handles_offline_unauthorized_and_multiple(self):
        for inventory in ([], [{'serial':'a','state':'unauthorized'}],
                          [{'serial':'a','state':'device'}, {'serial':'b','state':'device'}]):
            with self.assertRaises(RuntimeError):choose_serial(inventory)
        self.assertEqual(choose_serial([{'serial':'a','state':'device'}]), 'a')

    def test_adapter_checks_text_then_clicks_current_ui_bounds(self):
        class Field:
            info = {'bounds': {'left':80,'top':970,'right':1000,'bottom':1120}}
            value = ''
            def set_text(self, text, timeout):self.value = text
            def get_text(self, timeout):return self.value
        field = Field()
        class Objects:
            count = 1
            def __getitem__(self, i):return field
        class Client:
            clicks = []
            def __call__(self, **query):return Objects()
            def click(self, x, y):self.clicks.append((x,y))
        adapter = Android.__new__(Android)
        adapter.d = Client()
        adapter.capture = lambda:screen(code=field.value)
        self.assertTrue(adapter.submit(inspect(screen(), PROFILE), '001234', PROFILE))
        self.assertEqual(adapter.d.clicks, [(540,1970)])


if __name__ == '__main__':unittest.main()
