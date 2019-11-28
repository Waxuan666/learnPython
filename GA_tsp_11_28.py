"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np
from turtle import *
import folium  # 匯入 folium 套件
import webbrowser

import tkinter as tk
from tkinter import ttk

#  smtp发邮件
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

my_sender = '1173757227@qq.com'  # 发件人邮箱账号
my_pass = 'ayeckqwmugdogbcd'  # 发件人邮箱密码
my_user = 'waxuan666@gmail.com'  # 收件人邮箱账号


def mail():

    msg = MIMEMultipart()



    msg.attach(MIMEText('已将导航结果发至您的邮箱，请查看附件。', 'plain', 'utf-8'))
    msg['From'] = formataddr(["1173757227", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
    msg['To'] = formataddr(["waxuan", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
    msg['Subject'] = "导航结果"  # 邮件的主题，也可以说是标题

    att1 = MIMEApplication(open('map1.html', 'rb').read())

    att1.add_header('Content-Disposition', 'attachment', filename="map1.html")
    msg.attach(att1)

    server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
    server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
    server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
    server.quit()  # 关闭连接


#########################################################################################################


N_CITIES = 10  # DNA size
CROSS_RATE = 0.2
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 20
cityposition = np.array([[39.877781, 116.473008],
                         [39.872555, 116.475828],
                         [39.871441, 116.479883],
                         [39.873333, 116.472059],
                         [39.872748, 116.473609],
                         [39.877829, 116.475938],
                         [39.876627, 116.475342],
                         [39.872464, 116.475139],
                         [39.871292, 116.47743],
                         [39.877433, 116.474534]])


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        # 随机生成DNA，条件：0开头
        self.pop = np.vstack([np.random.permutation(range(DNA_size)) for _ in range(pop_size)])   ## 生成人口 DNA 矩阵，
        # self.pop = np.vstack([np.insert([np.random.permutation(
            # range(1, DNA_size))], 0, 0) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        # 根据DNA（0，1，2，3，4，5，6，7，8...）生成一个空数组矩阵
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)  # 同上
        for i, d in enumerate(DNA): # i是索引， d是城市编号组
            #d.astype('int64')
            city_coord = city_position[d.astype('int64')]    # 获得按照d排列的城市坐标组
            line_x[i, :] = city_coord[:, 0]  # 把横坐标放进数组
            line_y[i, :] = city_coord[:, 1]  # 把纵坐标放进数组
        return line_x, line_y                # 返回横纵坐标两个数组矩阵

    def get_fitness(self, line_x, line_y):
        last=new_size()
        '''总距离'''
        total_distance = np.empty(
            (line_x.shape[0],), dtype=np.float64)  # 创建一个空的数组矩阵
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):  # 将横纵坐标打包存为元组
            total_distance[i] = np.sum(
                np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))+ np.sqrt(np.square(line_x[i,0]-line_x[i,last-1])+np.square(line_y[i,0]-line_y[i,last-1]))  # numpy计算距离和(100个)
        # fitness = np.exp(self.DNA_size * 2 / total_distance) # 放大fitness
        fitness = 2**(6 - total_distance * 100)  # fitness是数组！

        return fitness, total_distance  # 返回

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size,
                               replace=True, p=fitness / fitness.sum())  # 高级！！！fitness越大，被选作后代的概率越大
        return self.pop[idx]  # 按照idx选出的新一代（有重复个体，而且总路径短的比例大，smart！）

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]                                       # find the city number
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent   

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent = child
        self.pop = pop


class TravelSalesPerson(object):
    # def __init__(self, n_cities):
    def __init__(self, cityposition):
        # self.city_position = np.random.rand(n_cities, 2) # 随机生成城市坐标二维数组
        self.city_position = cityposition
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        cx = self.city_position[:, 0].T
        cy = self.city_position[:, 1].T
        plt.scatter(cx, cy, s=100, c='k')
        for i in range(0, len(self.city_position)):
            #r'$（%s，%s）$' % self.city_position[i, 0], self.city_position[j, 1]
            plt.text(self.city_position[i, 0] + 1, self.city_position[i, 1], "(%s, %s)" % (
                self.city_position[i, 0], self.city_position[i, 1]), fontdict={'size': 5, 'color': 'red'})
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(39.874, 116.478, "Total distance=%.8f" %
                 total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((39.87, 39.88))
        plt.ylim((116.47, 116.48))
        plt.pause(0.001)



def run(cityposition1,NEW_SIZE):
    ga = GA(DNA_size=NEW_SIZE, cross_rate=CROSS_RATE,
            mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

    env = TravelSalesPerson(cityposition1)
    ax = plt.figure(1)
    for generation in range(N_GENERATIONS):  # 主循环 100代
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)  # 返回最大的fitness对应的索引
        print('Gen:', generation, '| best fit: %.8f' % fitness[best_idx],)

        env.plotting(np.append(lx[best_idx], lx[best_idx][0]), np.append(ly[best_idx], ly[best_idx][0]), total_distance[best_idx])  # 数据可视化
    #最终结果（天选之子）

    #Lx = lx[best_idx]
    #Ly = ly[best_idx]
    Lx = np.append(lx[best_idx], lx[best_idx][0])
    Ly = np.append(ly[best_idx], ly[best_idx][0])
    result = list(zip(Lx, Ly))
    plt.ioff()
    plt.pause(1)
    plt.close(ax)
#result=run.result



# 在地图中显示标记连线
    fmap = folium.Map(location=[39.8745, 116.4756], zoom_start=16)
    for i in range(len(result)):
        m = folium.Marker(result[i],
                          popup='<b>Skytree</b>')
        fmap.add_child(child=m)

    fmap.add_child(folium.PolyLine(locations=result,  # 坐标List
                                   weight=8))  # 线条宽度
    fmap.save('map1.html')

# webbrowser.open("file:///Users/wangyaxuan/Desktop/matplotlip/map1.html")

'''窗口页面设置'''
window = tk.Tk()
window.title('校园导游')
window.geometry('500x300')
l = tk.Label(window, text='你好！欢迎使用校园导游系统！', bg='orange', font=('Arial', 12), width=30, height=2)
l.pack()
l1 = tk.Label(window, text='请您选择目标位置', bg='orange', font=('Arial', 12), width=30, height=2)
l1.pack()

frame = tk.Frame(window)
frame.pack()

frame_select = tk.Frame(frame)
frame_select.pack()

frame_select1 = tk.Frame(frame)
frame_select1.pack()

btarray = np.array([0,1,2,3,4,5,6,7,8,9])
btvalue = []
for i in range(10):
    btvalue.append(tk.IntVar())
#position = []
def callbtValue():

    position = []
    for i in range(10):
        position.append(btvalue[i].get())
    print('已选中位置:', position)
    return position

c0 = ttk.Checkbutton(frame_select, text='宿舍', variable=btvalue[0], onvalue=1, offvalue=0,command=callbtValue)# 传值原理类似于radiobutton部件
c1 = ttk.Checkbutton(frame_select, text='奥运餐厅', variable=btvalue[1], onvalue=1, offvalue=0,command=callbtValue)
c2 = ttk.Checkbutton(frame_select, text='计算机中心', variable=btvalue[2], onvalue=1, offvalue=0,command=callbtValue)
c3 = ttk.Checkbutton(frame_select, text='信息楼', variable=btvalue[3], onvalue=1, offvalue=0,command=callbtValue)
c4 = ttk.Checkbutton(frame_select, text='图书馆', variable=btvalue[4], onvalue=1, offvalue=0,command=callbtValue)
c5 = ttk.Checkbutton(frame_select, text='校医院', variable=btvalue[5], onvalue=1, offvalue=0,command=callbtValue)
c6 = ttk.Checkbutton(frame_select, text='ATM1', variable=btvalue[6], onvalue=1, offvalue=0,command=callbtValue)
c7 = ttk.Checkbutton(frame_select, text='ATM2', variable=btvalue[7], onvalue=1, offvalue=0,command=callbtValue)
c8 = ttk.Checkbutton(frame_select, text='羽毛球馆', variable=btvalue[8], onvalue=1, offvalue=0,command=callbtValue)
c9 = ttk.Checkbutton(frame_select, text='回民餐厅', variable=btvalue[9], onvalue=1, offvalue=0,command=callbtValue)

c0.grid(row=0,column=0)
c1.grid(row=0,column=1)
c2.grid(row=0,column=2)
c3.grid(row=0,column=3)
c4.grid(row=0,column=4)
c5.grid(row=1,column=0)
c6.grid(row=1,column=1)
c7.grid(row=1,column=2)
c8.grid(row=1,column=3)
c9.grid(row=1,column=4)


#select_position = np.array(position)

#cityposition1 = np.array(cityposition[np.where(select_position == 1)])
#cityposition.astype('int64')



def new_size():
    newsize=0
    for i in range(10):
        newsize=newsize+btvalue[i].get()
    return newsize



def canrun():


    select_position = np.array(callbtValue())

    cityposition1 = np.array(cityposition[np.where(select_position == 1)])

    NEW_SIZE = btvalue[0].get() + btvalue[1].get() + btvalue[2].get() + btvalue[3].get() + btvalue[4].get() + btvalue[
        5].get() + btvalue[6].get() + btvalue[7].get() + btvalue[8].get() + btvalue[9].get()
    if new_size() >= 5:  # 如果选中5个以上
        print(NEW_SIZE)
        print(cityposition1)
        run(cityposition1,NEW_SIZE)
    else:
        tk.messagebox.showinfo(title='提示', message='请至少选择5项！')


def openhtml1():
    webbrowser.open("file:///Users/wangyaxuan/Desktop/matplotlip/map1.html")

def openhtml2():
    webbrowser.open("http://www.dianping.com/shop/91072974")


b0 = ttk.Button(frame_select1, text='运行', width=10,command=canrun)
b1 = ttk.Button(frame_select1, text='查看结果', width=10,command=openhtml1)
b2 = ttk.Button(frame_select1, text='发送至邮箱', width=10,command=mail)
b3 = ttk.Button(frame_select1, text='预定羽毛球馆',width=10,command=openhtml2)
b0.grid(row=0,column=0)
b1.grid(row=0,column=1)
b2.grid(row=2,column=0)
b3.grid(row=2,column=1)
window.mainloop()


