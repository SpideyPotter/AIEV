"""
Frame Capture for Panoptic Segmentation Demo (IronPython/.NET version)
Uses .NET APIs only (no requests/Pillow/mss/io dependencies).

This script:
1) Captures primary screen
2) Sends JPEG frame to FastAPI /segment endpoint
3) Parses JSON response (overlay_png_base64, latency_ms, segments)
4) Optionally saves overlay PNG to disk for debugging
"""
import sys
import time

import clr
clr.AddReference("System")
clr.AddReference("System.Net.Http")
clr.AddReference("System.Drawing")
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Web.Extensions")

import System
from System import Byte, Array
from System.IO import MemoryStream, File, FileMode, FileAccess
from System.Net.Http import HttpClient, MultipartFormDataContent, ByteArrayContent
from System.Drawing import Bitmap, Graphics
from System.Drawing.Imaging import ImageFormat
from System.Web.Script.Serialization import JavaScriptSerializer
from System.Windows.Forms import Screen

try:
    from GTA.UI import Notification
except Exception:
    Notification = None

# ===== CONFIGURATION =====
MODEL_HOST = "192.168.1.100"   # change to your headnode/worker LAN IP
MODEL_PORT = 8000
SEGMENT_ENDPOINT = "/segment"
FRAME_INTERVAL = 0.20          # 5 FPS
FRAME_WIDTH = 1024
FRAME_HEIGHT = 1024
REQUEST_TIMEOUT_SECONDS = 4.0

HEARTBEAT_FILE = "gta_panoptic_heartbeat.txt"
HEARTBEAT_INTERVAL = 2.0
NOTIFY_INTERVAL = 8.0
LOG_EVERY_N_FRAMES = 10

# Save latest overlay PNG returned by server (for debug)
SAVE_OVERLAY = True
OVERLAY_PATH = "gta_panoptic_overlay_latest.png"

# ===== GLOBALS =====
last_capture_time = 0.0
last_heartbeat_time = 0.0
last_notify_time = 0.0
frame_count = 0
ok_count = 0
err_count = 0
last_latency_ms = -1
last_segments_count = -1

http = HttpClient()
http.Timeout = System.TimeSpan.FromSeconds(REQUEST_TIMEOUT_SECONDS)
json = JavaScriptSerializer()
service_url = "http://{}:{}{}".format(MODEL_HOST, MODEL_PORT, SEGMENT_ENDPOINT)


def log(msg):
    print("[PANOPTIC DEMO] {}".format(msg))


def notify(msg):
    try:
        if Notification is not None:
            Notification.PostTicker(msg, True)
    except Exception:
        pass


def write_heartbeat(status, extra=""):
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            f.write("status={}\n".format(status))
            f.write("frame_count={}\n".format(frame_count))
            f.write("ok_count={}\n".format(ok_count))
            f.write("err_count={}\n".format(err_count))
            f.write("last_latency_ms={}\n".format(last_latency_ms))
            f.write("last_segments_count={}\n".format(last_segments_count))
            f.write("time={}\n".format(time.time()))
            f.write("service_url={}\n".format(service_url))
            if extra:
                f.write("extra={}\n".format(extra))
    except Exception:
        pass


def capture_jpeg_bytes():
    """Capture primary screen and return JPEG bytes via .NET stream."""
    screen = Screen.PrimaryScreen.Bounds
    bmp = Bitmap(screen.Width, screen.Height)
    gfx = Graphics.FromImage(bmp)
    ms = MemoryStream()
    try:
        gfx.CopyFromScreen(screen.X, screen.Y, 0, 0, screen.Size)
        resized = Bitmap(bmp, FRAME_WIDTH, FRAME_HEIGHT)
        try:
            resized.Save(ms, ImageFormat.Jpeg)
            return ms.ToArray()
        finally:
            resized.Dispose()
    finally:
        ms.Dispose()
        gfx.Dispose()
        bmp.Dispose()


def send_to_model(jpeg_bytes):
    """Send JPEG bytes to model server as multipart/form-data."""
    content = MultipartFormDataContent()
    byte_content = ByteArrayContent(Array[Byte](jpeg_bytes))
    try:
        content.Add(byte_content, "image", "frame.jpg")
        response = http.PostAsync(service_url, content).Result
        status_code = int(response.StatusCode)
        body = response.Content.ReadAsStringAsync().Result if response.Content is not None else ""
        return status_code, body
    finally:
        byte_content.Dispose()
        content.Dispose()


def decode_json_payload(body):
    """Parse JSON body into Python/.NET dictionary."""
    if body is None or body == "":
        return None
    try:
        return json.DeserializeObject(body)
    except Exception:
        return None


def save_overlay_from_base64(b64_text):
    """Decode base64 PNG and write to OVERLAY_PATH using .NET APIs."""
    if not b64_text:
        return False
    try:
        overlay_bytes = System.Convert.FromBase64String(b64_text)
        fs = File.Open(OVERLAY_PATH, FileMode.Create, FileAccess.Write)
        try:
            fs.Write(overlay_bytes, 0, overlay_bytes.Length)
            fs.Flush()
            return True
        finally:
            fs.Dispose()
    except Exception:
        return False


def handle_success(payload):
    global ok_count, last_latency_ms, last_segments_count
    ok_count += 1

    latency = -1
    seg_count = -1
    overlay_saved = False

    try:
        if payload is not None and "latency_ms" in payload:
            latency = int(payload["latency_ms"])
    except Exception:
        latency = -1

    try:
        if payload is not None and "segments" in payload and payload["segments"] is not None:
            seg_count = int(len(payload["segments"]))
    except Exception:
        seg_count = -1

    if SAVE_OVERLAY:
        try:
            overlay_b64 = payload["overlay_png_base64"] if payload is not None and "overlay_png_base64" in payload else None
            overlay_saved = save_overlay_from_base64(overlay_b64)
        except Exception:
            overlay_saved = False

    last_latency_ms = latency
    last_segments_count = seg_count

    if frame_count % LOG_EVERY_N_FRAMES == 0:
        log("Frame {} OK | latency={} ms | segments={} | overlay_saved={}".format(
            frame_count, latency, seg_count, overlay_saved
        ))


def tick():
    global last_capture_time, last_heartbeat_time, last_notify_time, frame_count, err_count
    now = time.time()

    if now - last_capture_time < FRAME_INTERVAL:
        return
    last_capture_time = now
    frame_count += 1

    if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
        write_heartbeat("running")
        last_heartbeat_time = now

    try:
        jpeg = capture_jpeg_bytes()
        status, body = send_to_model(jpeg)
        if status == 200:
            payload = decode_json_payload(body)
            handle_success(payload)
        else:
            err_count += 1
            log("Frame {}: model HTTP {}".format(frame_count, status))
            write_heartbeat("http_error", "status={}".format(status))
            if now - last_notify_time >= NOTIFY_INTERVAL:
                notify("[PANOPTIC] Model HTTP {} (check server)".format(status))
                last_notify_time = now
    except Exception as e:
        err_count += 1
        log("Frame {}: error {}".format(frame_count, e))
        write_heartbeat("exception", "err={}".format(e))
        if now - last_notify_time >= NOTIFY_INTERVAL:
            notify("[PANOPTIC] Frame send error (see log)")
            last_notify_time = now


def on_tick():
    tick()


write_heartbeat("script_loaded", "service_url={}".format(service_url))
log("Python executable: {}".format(getattr(sys, "executable", "unknown")))
log("Python version: {}".format(getattr(sys, "version", "unknown")))
log("Script loaded. Connecting to {}:{}".format(MODEL_HOST, MODEL_PORT))
log("Endpoint: {}".format(service_url))
log("If no response, check port-forward / firewall / MODEL_HOST.")
