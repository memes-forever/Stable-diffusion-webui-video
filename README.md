# Stable diffusion webui video

This script is created for use in the project https://github.com/AUTOMATIC1111/stable-diffusion-webui

using img2img generates pictures one after another

# Example



https://user-images.githubusercontent.com/107195976/191963848-6eb8e169-dee3-46d7-8310-db45046353fd.mp4


https://user-images.githubusercontent.com/107195976/191579132-f6f1c2e8-ba3d-4caf-aac4-f48ff4928516.mp4


https://user-images.githubusercontent.com/107195976/191961159-c878aca6-267d-4f35-afb9-2b1908a620c8.mp4



# Install
copy video.py in stable-diffusion-webui/scripts folder

# Using
1. create a picture or upload (it will be used as the first frame)
2. write a prompt
3. write end prompt
4. select video settings
5. run

# Usage

![image](https://user-images.githubusercontent.com/107195976/191533315-b09e0e08-ec0c-4a86-a1fc-c451438a4e98.png)

Extended Version:
![image](https://user-images.githubusercontent.com/12010863/195175113-12df7e14-4f96-4737-99c5-873b2278c796.png)


1.) End Prompt Trigger
Lets you define at how much Percent 0-100 the End Prompt will added to the original prompt

2.)Zoom Rotate
When zoom is activated you can select to rotate the image per rendered frame. Values from - to + 3.6° are accepted. (sanity limit else you get dark corners)

3.)TranslateXY
shifts the image in X and Y directions. check boxes if you want to go the opposite direction.
 if its tiled it takes the opposite end and copies it at the end of the scrolling direction
if not it does some color palette maintaining noise stretchy stuff at the end which works but is kind of hacky. 
Numpy Expert anyone? (would be good to keep the color palette intact)   





