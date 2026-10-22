"""uiautomator2 adapter for the supplied ADB workflow; local CPU inference only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from statistics import median

from .data import ROOT, save_json
from .phone import Profile, Screen, Watcher, inspect, same_image, signature


def device_inventory():
    import adbutils
    return [{'serial': d.serial, 'state': d.state} for d in adbutils.adb.list()]


def choose_serial(inventory, requested=None):
    if requested:
        selected = [d for d in inventory if d['serial'] == requested]
        if not selected:
            raise RuntimeError('Указанный телефон отсутствует в ADB. Проверьте USB и отладку.')
        if selected[0]['state'] != 'device':
            raise RuntimeError('Телефон недоступен: ' + selected[0]['state'] +
                               '. Разблокируйте экран и разрешите отладку по USB.')
        return requested
    if len(inventory) != 1:
        raise RuntimeError('ADB: устройств ' + str(len(inventory)) +
                           '. Подключите один телефон или укажите --serial.')
    return choose_serial(inventory, inventory[0]['serial'])


class Android:
    def __init__(self, serial):
        import uiautomator2 as u2
        self.d = u2.connect_usb(serial)
        self.d.implicitly_wait(1)

    def capture(self):
        package = self.d.app_current()['package']
        hierarchy = self.d.dump_hierarchy()
        image = self.d.screenshot()
        if self.d.app_current()['package'] != package:
            raise RuntimeError('Приложение сменилось во время чтения экрана')
        if image is None or image.width < 2 or image.height < 2:
            raise RuntimeError('ADB не вернул изображение экрана')
        return Screen(image.convert('RGB'), hierarchy, package)

    def submit(self, view, code, profile):
        field = view.field
        if not field:
            return False
        query = {'className': field.class_name}
        if field.resource_id:
            query['resourceId'] = field.resource_id
        objects = self.d(**query)
        selected = []
        for index in range(objects.count):
            obj = objects[index]
            b = obj.info['bounds']
            if (b['left'], b['top'], b['right'], b['bottom']) == field.bounds:
                selected.append(obj)
        if len(selected) != 1:
            return False
        selected[0].set_text(code, timeout=1)
        if selected[0].get_text(timeout=1) != code:
            return False
        # No Back key or fixed tap coordinates. Re-read controls after the UI updates.
        for _ in range(3):
            current = inspect(self.capture(), profile)
            if current.state != 'ready' or not same_image(signature(view.crop), signature(current.crop)):
                return False
            if current.field.text != code:
                return False
            if current.button:
                x1, y1, x2, y2 = current.button.bounds
                self.d.click((x1+x2)//2, (y1+y2)//2)
                return True
            time.sleep(.2)
        return False

    def refresh(self, view, profile):
        current = inspect(self.capture(), profile)
        if (current.state != 'ready' or not current.refresh
                or not same_image(signature(view.crop), signature(current.crop))):
            return False
        x1, y1, x2, y2 = current.refresh.bounds
        self.d.click((x1+x2)//2, (y1+y2)//2)
        return True


def connect(serial=None):
    return Android(choose_serial(device_inventory(), serial))


def capture(serial, output):
    screen = connect(serial).capture()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    screen.image.save(folder/'screen.png')
    (folder/'ui.xml').write_text(screen.hierarchy, encoding='utf-8')
    save_json(folder/'device.json', {'package': screen.package, 'size': screen.image.size})
    print('Экран и дерево интерфейса сохранены локально:', folder.resolve())


def watch(serial, profile_path, *, observe=False, duration=0, output=None):
    from .model import load_model
    profile = Profile.load(profile_path)
    device = connect(serial)  # Check the phone before loading/downloading the model.
    predictor = load_model()
    folder = Path(output or ROOT/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=True)
    print('Наблюдение. Ctrl+C — остановить. Журнал:', folder.resolve())
    with (folder/'events.jsonl').open('a', encoding='utf-8') as log:
        def emit(**event):
            event['time'] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            log.write(line+'\n')
            log.flush()
        watcher = Watcher(device, predictor, profile, observe=observe, emit=emit)
        started, errors = time.monotonic(), 0
        durations = []
        frames = 0
        try:
            while not duration or time.monotonic()-started < duration:
                frame_start = time.monotonic()
                try:
                    watcher.tick()
                    errors = 0
                    frames += 1
                    durations.append((time.monotonic()-frame_start)*1000)
                except Exception as exc:
                    errors += 1
                    # Broken/empty UI does not count as an absent challenge.
                    watcher.absent = watcher.stable = 0
                    watcher.previous = None
                    emit(status='read_or_action_error', kind=type(exc).__name__, consecutive=errors)
                    if errors >= 3:
                        raise RuntimeError('Три ошибки связи/интерфейса подряд; наблюдение остановлено') from exc
                time.sleep(profile.poll_seconds)
        except KeyboardInterrupt:
            emit(status='stopped')
        finally:
            emit(status='session_end', frames=frames, elapsed_seconds=time.monotonic()-started,
                 median_iteration_ms=median(durations) if durations else None,
                 observe=observe, attempts=watcher.attempts)


def evaluate_phone():
    """Actual inference on two separately labeled Drive development cases."""
    import csv
    import hashlib
    from PIL import Image
    from .model import load_model
    predictor = load_model()
    predictor(Image.new('RGB', (160, 48), 'white'))
    rows = []
    with (ROOT/'datasets/phone/labels.csv').open(encoding='utf-8', newline='') as source:
        for item in csv.DictReader(source):
            path = ROOT/item['image_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Phone development image checksum mismatch')
            start = time.perf_counter()
            with Image.open(path) as image:
                if item['roi']:
                    box = json.loads(item['roi'])
                    image = image.crop(tuple(int(v*(image.width if i%2 == 0 else image.height))
                                             for i, v in enumerate(box)))
                raw, score = predictor(image)
            rows.append(dict(item, prediction=raw, score=score, correct=raw == item['label'],
                             elapsed_ms=(time.perf_counter()-start)*1000))
    result = dict(dataset='phone_development', n=len(rows), correct=sum(r['correct'] for r in rows),
                  label_source='assistant visual transcription, not independent annotation',
                  hardware_verified=False, paid_calls=0, rows=rows)
    save_json(ROOT/'research/results/phone.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
