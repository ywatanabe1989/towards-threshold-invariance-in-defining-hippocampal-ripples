import matplotlib.pyplot as plt

def rgb2bgr(rgb_colors):
    return (rgb_colors[1],
            rgb_colors[0],
            rgb_colors[2])

def get_color_code(color_name, use_rgb=True):
    RGB_PALLETE = {
        'blue':(0,128,192),
        'red':(255,70,50),
        'pink':(255,150,200),
        'green':(20,180,20),
        'yellow':(230,160,20),
        'glay':(128,128,128),
        'parple':(200,50,255),
        'light_blue':(20,200,200),
        'blown':(128,0,0),
        'navy':(0,0,100),
    }

    if use_rgb == True: # RGB
        color_code = RGB_PALLETE[color_name]
    else: # BGR
        color_code = rgb2bgr(RGB_PALLETE[color_name])

    return color_code

def get_plot_color(color_name):
    color_code = get_color_code(color_name, use_rgb=True)
    return np.array(color_code) / 255



if __name__ == '__main__':
    ## DEMO
    th = np.linspace(0, 2*np.pi, 128)
    plt.plot(th, np.cos(th), color=get_plot_color('green'))
