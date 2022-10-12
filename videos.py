import numpy as np
from tqdm import trange
import glob
import os
import modules.scripts as scripts
import gradio as gr
import subprocess
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from PIL import Image


class Script(scripts.Script):
    def title(self):
        return "Videos"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        prompt_end_trigger=gr.Slider(minimum=0.0, maximum=0.9, step=0.1, label='End Prompt Blend Trigger Percent', value=0)
        prompt_end = gr.Textbox(label='Prompt end', value="")

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
        return [ show, prompt_end,prompt_end_trigger, seconds, fps, smooth, denoising_strength_change_factor, zoom, zoom_level,
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


    def run(self, p,  show,
            prompt_end, prompt_end_trigger, seconds, fps, smooth, denoising_strength_change_factor, zoom, zoom_level,
            direction_x, direction_y, rotate, degree,is_tiled, trnx,trnx_left, trnx_percent,trny,trny_up,trny_percent):  # , denoising_strength_change_factor
        processing.fix_seed(p)

        p.batch_size = 1
        p.n_iter = 1

        batch_count = p.n_iter
        p.extra_generation_params = {
            "Denoising strength change factor": denoising_strength_change_factor,
        }

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        loops = seconds * fps

        grids = []
        all_images = []
        state.job_count = loops * batch_count

        # fifty = int(loops/2)

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        for n in range(batch_count):
            history = []

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True
#TODO: Hook in here and use ffmpeg to directly make a movie. only safe certain keyframes to have some kind of preview.
                if opts.img2img_color_correction:
                    p.color_corrections = initial_color_corrections
                p.color_corrections = initial_color_corrections
                
                if i > int(loops*prompt_end_trigger) and prompt_end not in p.prompt and prompt_end != '':
                    p.prompt = prompt_end.strip() + ' ' + p.prompt.strip()

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

                if i == 0:
                    # First image
                    init_img = p.init_images[0]
                    seed = p.seed
                    images.save_image(init_img, p.outpath_samples, "", seed, p.prompt)
                else:
                    processed = processing.process_images(p)
                    init_img = processed.images[0]
                    seed = processed.seed

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
                history.append(init_img)

            grid = images.image_grid(history, rows=1)
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info,
                                 short_filename=not opts.grid_extended_filename, grid=True, p=p)

            grids.append(grid)
            all_images += history

        if opts.return_grid:
            all_images = grids + all_images

        processed = Processed(p, all_images if show else [], initial_seed, initial_info)

        files = [i for i in glob.glob(f'{p.outpath_samples}/*.png')]
        files.sort(key=lambda f: os.path.getmtime(f))
        files = files[-loops:]
        files = files + [files[-1]]  # minterpolate smooth break last frame, dupplicate this

        video_name = files[-1].split('\\')[-1].split('.')[0] + '.mp4'

        video_path = make_video_ffmpeg(video_name, files=files, fps=fps, smooth=smooth)
        play_video_ffmpeg(video_path)
        processed.info = processed.info + '\nvideo save in ' + video_path

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
def ffmpeg_are_you_there(path): 
    ffmpeg_dir = os.path.join(path, 'ffmpeg')
    result = "no"
    #is ffmpeg in da path?
    try:
        subprocess.call(["ffmpeg", "--version"])
        result = "yes"
    except OSError as e:
        if e.errno == errno.ENOENT and os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, 'ffmpeg.exe'))):
            result = "installed"
    return result
            


def make_video_ffmpeg(video_name, files=[], fps=10, smooth=True):
    import modules
    path = modules.paths.script_path
    save_dir = 'outputs/img2img-videos/'
    is_ffmpeg_already_in_path = ffmpeg_are_you_there(path)
    if  is_ffmpeg_already_in_path == "installed" or is_ffmpeg_already_in_path == "yes":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        install_ffmpeg(path, save_dir)

    ffmpegstr = 'ffmpeg/ffmpeg'
    if is_ffmpeg_already_in_path == "yes":
            ffmpegstr = 'ffmpeg'

    video_name = save_dir + video_name
    txt_name = 'list.txt'

    # save pics path in txt
    open(txt_name, 'w').write('\n'.join(["file '" + os.path.join(path, f) + "'" for f in files]))

    # -vf "tblend=average,framestep=1,setpts=0.50*PTS"
    subprocess.call(' '.join([
        ffmmpegstr,' -y',
        f'-r {fps}',
        '-f concat -safe 0',
        f'-i "{txt_name}"',
        '-vcodec libx264',
        '-filter:v minterpolate' if smooth else '',   # smooth between images
        '-crf 10',
        '-pix_fmt yuv420p',
        f'"{video_name}"'
    ]))
    return video_name


def play_video_ffmpeg(video_path):
    ffplaystr="ffplay"
    try:
        subprocess.call(["ffplay", "--version"])
        result = "yes"
    except OSError as e:
        if e.errno == errno.ENOENT:
            ffplaystr = "ffmpeg/ffplay"

    subprocess.Popen(f'''"{ffplaystr}" "{video_path}"''')
    