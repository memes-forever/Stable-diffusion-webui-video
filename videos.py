import numpy as np
from tqdm import trange
import glob
import os
import sys
import modules.scripts as scripts
import gradio as gr
import subprocess
from subprocess import Popen, PIPE
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from PIL import Image


####Cooldown HACK
from time import time
from time import sleep
import wmi #Open Hardware Monitor
#https://openhardwaremonitor.org/
#makes total sense to overwrite the gpu fan power to 100% before starting to render.
#im using https://www.msi.com/Landing/afterburner/graphics-cards for that

#not sure if this should be ui configurable.
TEMP_MAX = 88
TEMP_MIN = 64
#decreased rendering time for 30s from 2h:19m to 53m


class Script(scripts.Script):
    def title(self):
        return "Img2Video"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        outputname = gr.Textbox(label="Output Name", lines=1)
        prompt_end_trigger=gr.Slider(minimum=0.0, maximum=0.9, step=0.1, label='End Prompt Blend Trigger Percent', value=0)
        prompt_end = gr.Textbox(label='Prompt end', value="")

        previews = gr.Checkbox(label='Save Previews', value=True)
        smooth = gr.Checkbox(label='Smooth video', value=True)
        seconds = gr.Slider(minimum=1, maximum=250, step=1, label='Seconds', value=1)
        fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=10)

        denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01,
                                                     label='Denoising strength change factor', value=1)

        zoom = gr.Checkbox(label='Zoom', value=False)
        zoom_level = gr.Slider(minimum=1, maximum=1.1, step=0.001, label='Zoom level', value=1)
        direction_x = gr.Slider(minimum=-0.1, maximum=0.1, step=0.01, label='Direction X', value=0)
        direction_y = gr.Slider(minimum=-0.1, maximum=0.1, step=0.01, label='Direction Y', value=0)
        rotate = gr.Checkbox(label='Rotate', value=False)
        degree = gr.Slider(minimum=-3.6, maximum=3.6, step=0.1, label='Degrees', value=0)
        is_tiled = gr.Checkbox(label="Is the Image Tiled?", value = False)
        trnx =  gr.Checkbox(label='TranslateX', value=False)
        trnx_left =  gr.Checkbox(label='Left', value=False)
        trnx_percent = gr.Slider(minimum=0, maximum=50, step=1, label='PercentX', value=0)
        trny =  gr.Checkbox(label='TranslateY', value=False)
        trny_up =  gr.Checkbox(label='Up', value=False)
        trny_percent = gr.Slider(minimum=0, maximum=50, step=1, label='PercentY', value=0)
        show = gr.Checkbox(label='Show generated pictures in ui', value=False)
        return [outputname, show, smooth, prompt_end,prompt_end_trigger, seconds, fps, previews, denoising_strength_change_factor, zoom, zoom_level,
                direction_x, direction_y, rotate, degree,is_tiled,trnx,trnx_left,trnx_percent,trny,trny_up,trny_percent]

    def zoom_into(self, img, zoom, direction_x, direction_y):
        neg = lambda x: 1 if x > 0 else -1
        if abs(direction_x) > zoom-1:
            # *0.999999999999999 to avoid a float rounding error that makes it higher than desired
            direction_x = (zoom-1)*neg(direction_x)*0.999999999999999
        if abs(direction_y) > zoom-1:
            direction_y = (zoom-1)*neg(direction_y)*0.999999999999999
        w, h = img.size
        x = w/2+direction_x*w/4
        y = h/2-direction_y*h/4
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2,
                        x + w / zoom2, y + h / zoom2))
        return img.resize((w, h), Image.LANCZOS)

    def rotate(self,img:Image, degrees: float):
        img = img.rotate(degrees)
        return img

    def blend(self,signal,noisey):
        noisey = noisey[np.random.permutation(noisey.shape[0]),:,:]
        noisey= noisey[:,np.random.permutation(noisey.shape[1]),:]
        #TODO figure out how to do this in numpy i guess we can save time here. this runs with 32 ms.
        img_tmp = Image.fromarray(signal)
        noise = Image.fromarray(noisey)
        img_tmp.putalpha(1)
        noise.putalpha(1)
        blend = Image.blend(img_tmp, noise, 0.3)
        blend.convert("RGB")
        bg = Image.new("RGB", blend.size, (255, 255, 255) )
        bg.paste(blend)
        result = np.array(bg)
        return result


    def translateY(self,img:Image, percent : int, is_tiled: bool, up: bool = False):
        w,h = img.size
        scl = h*(percent/100.0)
        h = int(scl)
        na = np.array(img)
        print(na.shape)
        if up:
            nextup = na[0:h,:,:]
            nextdown = na[-h:, :,:]
            nextup = self.blend(nextup,nextdown)
            if is_tiled:
                nextup = na[ -h:,:,:]
            na = np.vstack((nextup,na))
            na=na[:-h, :]
        else:
            nextdown = na[-h:, :,:]
            nextup = na[0:h,:,:]
            nextdown = self.blend(nextdown,nextup)
            if is_tiled:
                nextdown = na[0:h, :,:]
            na=np.vstack((na,nextdown))
            na=na[h:,:] 
        img = Image.fromarray(na)
        return img

    def translateX(self,img:Image, percent : int, is_tiled: bool,  left: bool = False):
        w,h = img.size
        scl = w*(percent/100)
        w = int(scl)
        na = np.array(img)
        if left:
            nextleft = na[:,0:w:,:]
            nextright = na[:,-w:,:]
            nextleft = self.blend(nextleft,nextright)
            if is_tiled:
                nextleft = na[:,-w:,:]
            na=np.hstack((nextleft,na))
            na=na[:,:-w]
        else:
            nextright = na[:,-w:,:]
            nextleft = na[:,0:w:,:]
            nextright = self.blend(nextright,nextleft)
            if is_tiled:
                nextright = na[:,0:w,:]
            na=np.hstack((na, nextright))
            na=na[:,w:]       
        img = Image.fromarray(na)
        return img


    ### Cooldown HACK
    def measureGpuTemp(self):
        pythoncom.CoInitialize()
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        gpu_temp = 0
        for sensor in sensors:
            if  sensor.SensorType==u'Temperature' and 'GPU' in sensor.Name:
                gpu_temp = sensor.Value
        return gpu_temp

    def run(self, p, outputname,  show,
            prompt_end, prompt_end_trigger, seconds, fps, previews, denoising_strength_change_factor, zoom, zoom_level,
            direction_x, direction_y, rotate, degree,is_tiled, trnx,trnx_left, trnx_percent,trny,trny_up,trny_percent):  # , denoising_strength_change_factor
        processing.fix_seed(p)

        import modules
        p.batch_size = 1
        p.n_iter = 1

        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
        }

        output_images, info = None, None
        initial_seed = None
        initial_info = None
        if fps is None or fps < 10:
            fps=10
        loops = seconds * fps

        grids = []
        all_images = []
        state.job_count = loops * batch_count

        save_dir = 'outputs/img2img-video/'
        if outputname is None or outputname =="":
            outputname="output"
        output_file = outputname+".ts" #if something breaks no chance to recover mp4 file..
        # Taken from https://github.com/Filarius
        # Author: Filarius 
        encoder = ffmpeg(
            " ".join(
                [
                    "ffmpeg -y -loglevel panic",
                    "-f rawvideo -pix_fmt rgb24",
                    f"-s:v {p.width}x{p.height} -r {fps}",
                    "-i - -c:v libx264 -pix_fmt yuv420p -preset fast",
                    '-filter:v minterpolate' if smooth else '',
                    f'-crf 10 "{save_dir}/{output_file}"',
                ]
            ),
            use_stdin=True,
        )
        encoder.start()


        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
####Cooling HACK
        start = time()*10000 #seconds
        for n in range(batch_count):
            history = []
            loops = loops +1
            for ii in range(loops):


           
                ### Gpu Cooling HACK
                #Check for cooldown
                now = time()*10000 #seconds
                if start-now >=30:  
                    earlier = start 
                    start = now
                    gpu_temp = self.measureGpuTemp() #this operation takes 531ms  
                    print("\n gpu is at: " , gpu_temp, " °C .")
                    if gpu_temp >= TEMP_MAX :  #GPU Clock goes down here RTX 2080TI
                        while gpu_temp > TEMP_MIN: #ddunno. #letting fans run at 100% baseline is 35°
                            gpu_temp = self.measureGpuTemp()
                            sleep(10)
                            print("Waiting... Temp is now: " ,gpu_temp, " °C \n")
                        print("+++ Waited for: ", start-earlier, " seconds to CoolDown +++ \n")
            #####

                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True
                p.do_not_save_samples = True
                if ii % fps == 0 and preview: # 1 sample per second if preview enabled
                    p.do_not_save_samples = False
                p.color_corrections = initial_color_corrections
                
                if ii > int(loops*prompt_end_trigger) and prompt_end not in p.prompt and prompt_end != '':
                    p.prompt = prompt_end.strip() + ' ' + p.prompt.strip()

                state.job = f"Iteration {ii + 1}/{loops}, batch {n + 1}/{batch_count}"

                if ii == 0:
                    # First image
                    init_img = p.init_images[0]
                    seed = p.seed
                    images.save_image(init_img, p.outpath_samples, "", seed, p.prompt)
                else:
                    processed = processing.process_images(p)

                    init_img = processed.images[0]
                    seed = processed.seed
                    encoder.write(np.asarray(init_img))
                    if initial_seed is None:
                        initial_seed = processed.seed
                        initial_info = processed.info

                    if zoom and zoom_level != 1:
                        if rotate and degree != 0:
                            init_img=self.rotate(init_img, degree)
                        init_img = self.zoom_into(init_img, zoom_level, direction_x, direction_y)

                    if trnx and trnx_percent > 0:
                        init_img = self.translateX(init_img, trnx_percent, is_tiled, trnx_left)
                    
                    if trny and trny_percent > 0:
                        init_img = self.translateY(init_img, trny_percent, is_tiled, trny_up)

                p.init_images = [init_img]

                p.seed = seed + 1
                p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                 

        processed = Processed(p,  [], initial_seed, initial_info)

        return processed


def install_ffmpeg(path, save_dir):
    from basicsr.utils.download_util import load_file_from_url
    from zipfile import ZipFile

    ffmpeg_url = 'https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip'
    ffmpeg_dir = os.path.join(path, 'ffmpeg')

    ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)

    if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, 'ffmpeg.exe'))):
        with ZipFile(ckpt_path, 'r') as zipObj:
            listOfFileNames = zipObj.namelist()
            for fileName in listOfFileNames:
                if '/bin/' in fileName:
                    zipObj.extract(fileName, ffmpeg_dir)
        os.rename(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], 'bin', 'ffmpeg.exe'), os.path.join(ffmpeg_dir, 'ffmpeg.exe'))
        os.rename(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], 'bin', 'ffplay.exe'), os.path.join(ffmpeg_dir, 'ffplay.exe'))
        os.rename(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], 'bin', 'ffprobe.exe'), os.path.join(ffmpeg_dir, 'ffprobe.exe'))

        os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], 'bin'))
        os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
    os.makedirs(save_dir, exist_ok=True)
    return

#this typpe annotation syntax makes me happy.
def ffmpeg_are_you_there(save_dir :str): 
    import modules
    path = modules.paths.script_path
    result = False
    ffmpeg_dir = os.path.join(path, 'ffmpeg')
    #is ffmpeg in da path?
    try:
        subprocess.call(["ffmpeg", "--version"])
        result = True
    except OSError as e:
        if e.errno == errno.ENOENT and os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, 'ffmpeg.exe'))):
            result = True #Well its installed locally so we are fine
    return result
            
1
# Taken from https://github.com/Filarius
# Author: Filarius 
class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None