import shlex
import subprocess
import io

from . import printing
from . import timer

@timer.section('gif')
def encode_gif(frames, fps):
  # Collect frames into single byte buffer because Popen doesn't support multiple writes (writing to
  # proc.stdin is discouraged and caused us to fail to capture any error output).
  buffer = io.BytesIO()
  for frame in frames:
    buffer.write(frame.tobytes())
  buffer.seek(0)

  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-r', f'{fps:.02f}',
    '-s', f'{w}x{h}',
    '-pix_fmt', f'{pxfmt}',
    '-i', '-',
    '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
    '-r', f'{fps:.02f}',
    '-f', 'gif',
    '-',
  ]
  try:
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
      out, err = proc.communicate(input=buffer.getvalue())

      if proc.returncode:
        err_utf8 = err.decode('utf8')
        raise IOError(f'Failed to run {cmd}\n{err_utf8}')

      return out
  except Exception as e:
    printing.print_(f'Failed to run {shlex.join(cmd)}')
    raise e
