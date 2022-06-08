
from cgitb import text
from re import sub
import glfw
from compushady import get_discovered_devices, Buffer, HEAP_DEFAULT,HEAP_UPLOAD,HEAP_READBACK,Compute, Swapchain,Texture2D
from compushady.shaders import hlsl
import compushady.config
import compushady.formats as formats
import random
import struct
import numpy as np

from platform import system




def ball_packer(array):
    buffer = bytes(0)
    for item in array:
        buffer += struct.pack('8f', item.x, item.y, item.width, item.height, *item.color)
    return buffer

def collide(source,dest):

        if source.x+source.width + normalized_direction[0]*speed<dest.x:
            return False
        if source.x+normalized_direction[0]*speed>dest.x+dest.width:
            return False
        if source.y+source.height+normalized_direction[1]*speed<dest.y:
            return False
        if source.y+normalized_direction[1]*speed>dest.y+dest.height:
            return False
        return True


compushady.config.set_debug(True) #la gpu va 4 volte + lenta, quando ho finito di programmare va eliminato

texture =Texture2D(256,256,formats.B8G8R8A8_UNORM)


class Ball:
    def __init__(self,x,y,width,height,color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.is_alive=True

BLUE = [0,0,1,1]
BLACK = [0,0,0,1]
RED =  [1,0,0,1]
GREEN = [0,1,0,1]
YELLOW = [1,1,0,1]
WHITE = [1,1,1,1]
CYAN = [0,1,1,1]
INVISIBLE = [0,0,0,0]

colors=[BLUE,RED,GREEN,YELLOW,WHITE,CYAN]
index=0
is_pressed=False
list_of_drawables = []
list_of_balls = []

rows=4
cols = 20

start_x = 0
start_y = 0
for x in range (0,rows):
    for y in range (0,cols):
        ball=Ball(start_x,start_y,20,20,colors[random.randint(0,len(colors)-1)])
        start_x+=ball.width+1
        list_of_drawables.append(ball)
        list_of_balls.append(ball)  
    start_x=0
    start_y+=21

bullet = Ball(texture.width//2,texture.height-20,20,20,WHITE)
list_of_drawables.append(bullet)
bullet_direction = [0,0]
speed = 3
fire_position = [bullet.x,bullet.y]

balls_staging_buffer = Buffer(8*4*200,HEAP_UPLOAD)
balls_buffer = Buffer(
    balls_staging_buffer.size,format=formats.R32G32B32A32_SINT)


shader = hlsl.compile("""
struct Ball_s
{
    float2 pos;
    float2 size;
    float4 color;
};

StructuredBuffer<Ball_s> balls : register(t0);
RWTexture2D<float4> texture : register(u0);
[numthreads(8, 8, 16)]
void main(int3 tid : SV_DispatchThreadID)
{
    Ball_s ball = balls[tid.z];
    if(tid.x > ball.pos.x + ball.size.x)
        return;
    if (tid.x < ball.pos.x)
        return;
    if (tid.y < ball.pos.y)
        return;
    if (tid.y > ball.pos.y + ball.size.y)
        return;
    texture[tid.xy] = ball.color;
}
""")

compute = Compute(shader,srv=[balls_buffer],uav=[texture])


clear_screen = Compute(hlsl.compile("""
RWTexture2D<float4> texture : register(u0);
[numthreads(8, 8, 1)]
void main(int3 tid : SV_DispatchThreadID)
{
    texture[tid.xy] = float4(0, 0, 0, 0);
}
"""), uav=[texture])

glfw.init()
#we don't want implicit openGL
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
window = glfw.create_window(
    texture.width,texture.height,"FirstWindow",None,None)

if system() == 'Windows':
    swapchain = Swapchain(glfw.get_win32_window(
        window), formats.B8G8R8A8_UNORM, 2)
elif system() == 'Darwin':
    from compushady.backends.metal import create_metal_layer
    ca_metal_layer = create_metal_layer(glfw.get_cocoa_window(window), formats.B8G8R8A8_UNORM)
    swapchain = Swapchain(
        ca_metal_layer,formats.B8G8R8A8_UNORM, 2)
else:
    swapchain = Swapchain((glfw.get_x11_display(), glfw.get_x11_window(
        window)), formats.B8G8R8A8_UNORM, 2)

normalized_direction = [0,0]
while not glfw.window_should_close(window):
    glfw.poll_events()
    state = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
    if state == glfw.PRESS:
        bullet_direction=glfw.get_cursor_pos(window)
        bullet_pos = np.array([bullet.x,bullet.y])
        direction = np.subtract(bullet_direction,bullet_pos)
        normalized_direction = direction/np.sqrt(np.sum(direction**2))
        

    bullet.x += normalized_direction[0]*speed
    bullet.y += normalized_direction[1]*speed

    if glfw.get_key(window, glfw.KEY_A) and is_pressed == False:
        index-=1
        if index<0:
            index=len(colors)-1
        bullet.color=colors[index]


    index = 0
    for ball in list_of_balls:
        index+=1
        if collide(bullet,ball):
                normalized_direction[0] *= -1
                normalized_direction[1] *= -1
                if ball.color==bullet.color:
                    ball.color=INVISIBLE
                    list_of_balls.remove(ball)
                    list_of_drawables.remove(ball)
                    normalized_direction=[0,0]
                    bullet.x,bullet.y = fire_position
                    break
                else:
                    new_ball= Ball(bullet.x,bullet.y,bullet.width,bullet.height,bullet.color)
                    list_of_balls.append(new_ball)
                    list_of_drawables.append(new_ball)
                    normalized_direction=[0,0]
                    bullet.color = colors[random.randint(0,len(colors)-1)]
                    bullet.x,bullet.y = fire_position


    index=0
    clear_screen.dispatch(texture.width // 8, texture.height // 8, 1)
    
    #wall collision
    if bullet.x + bullet.width >= texture.width:
        normalized_direction[0] *= -1
    if bullet.x < 0:
        normalized_direction[0] *= -1
    if bullet.y + bullet.height >= texture.height:
        normalized_direction[1] *= -1
    if bullet.y < 0:
        normalized_direction[1] *= -1

    balls_staging_buffer.upload(ball_packer(list_of_drawables))
    balls_staging_buffer.copy_to(balls_buffer)

    compute.dispatch(texture.width // 8, texture.height // 8, len(list_of_drawables)//10)

    swapchain.present(texture)

swapchain = None  # this ensures the swapchain is destroyed before the window

glfw.terminate()
